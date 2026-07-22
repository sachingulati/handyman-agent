import sqlite3

import config
import db
import server


def test_gemma_delegate_rejects_missing_working_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    result = server.gemma_delegate("do a thing", str(tmp_path / "does-not-exist"))
    assert "error" in result


def test_gemma_delegate_spawns_when_under_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(config, "MAX_CONCURRENT_JOBS", 3)
    spawned = []
    monkeypatch.setattr(server, "_spawn_worker", lambda job_id: spawned.append(job_id))

    result = server.gemma_delegate("do a thing", str(tmp_path))

    assert result["status"] == "running"
    assert spawned == [result["job_id"]]


def test_gemma_delegate_queues_when_at_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(config, "MAX_CONCURRENT_JOBS", 1)
    spawned = []
    monkeypatch.setattr(server, "_spawn_worker", lambda job_id: spawned.append(job_id))

    first = server.gemma_delegate("first", str(tmp_path))
    second = server.gemma_delegate("second", str(tmp_path))

    assert first["status"] == "running"
    assert second["status"] == "queued"
    assert spawned == [first["job_id"]]


def test_gemma_check_returns_status_and_summary_when_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    conn = db.connect(config.DB_PATH)
    job_id = db.create_job(conn, "t", str(tmp_path))
    db.update_status(conn, job_id, "done", result_summary="all finished")
    conn.close()

    result = server.gemma_check(job_id)
    assert result["status"] == "done"
    assert result["result_summary"] == "all finished"


def test_gemma_check_omits_summary_when_running(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    conn = db.connect(config.DB_PATH)
    job_id = db.create_job(conn, "t", str(tmp_path))
    db.update_status(conn, job_id, "running")
    conn.close()

    result = server.gemma_check(job_id)
    assert result["status"] == "running"
    assert "result_summary" not in result


def test_gemma_check_unknown_job_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    db.connect(config.DB_PATH).close()
    result = server.gemma_check("nope")
    assert "error" in result


def test_gemma_cancel_sets_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    conn = db.connect(config.DB_PATH)
    job_id = db.create_job(conn, "t", str(tmp_path))
    conn.close()

    result = server.gemma_cancel(job_id)
    assert result["status"] == "cancel_requested"

    conn = db.connect(config.DB_PATH)
    assert db.is_cancel_requested(conn, job_id) is True


def test_gemma_cancel_unknown_job_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    db.connect(config.DB_PATH).close()
    result = server.gemma_cancel("nope")
    assert "error" in result


def test_gemma_delegate_marks_job_error_when_spawn_fails_after_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(config, "MAX_CONCURRENT_JOBS", 3)

    def _boom(job_id):
        raise OSError("failed to launch interpreter")

    monkeypatch.setattr(server, "_spawn_worker", _boom)

    result = server.gemma_delegate("do a thing", str(tmp_path))

    assert result["status"] == "error"
    job_id = result["job_id"]

    conn = db.connect(config.DB_PATH)
    job = db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "error"
    assert "failed to launch interpreter" in job["result_summary"]


def test_gemma_delegate_returns_error_dict_when_db_connect_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")

    def _boom(path):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "connect", _boom)

    result = server.gemma_delegate("do a thing", str(tmp_path))

    assert "error" in result
    assert "job_id" not in result


def test_gemma_check_returns_error_dict_when_db_connect_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")

    def _boom(path):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "connect", _boom)

    result = server.gemma_check("some-id")

    assert "error" in result


def test_gemma_cancel_returns_error_dict_when_db_connect_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")

    def _boom(path):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "connect", _boom)

    result = server.gemma_cancel("some-id")

    assert "error" in result
