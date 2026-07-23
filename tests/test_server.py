import sqlite3

from conftest import make_config
from handyman import config
from handyman import db
from handyman import server
def test_gemma_delegate_rejects_missing_working_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    result = server.gemma_delegate("do a thing", str(tmp_path / "does-not-exist"))
    assert "error" in result


def test_gemma_delegate_spawns_when_under_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db", max_concurrent_jobs=3))
    spawned = []
    monkeypatch.setattr(server, "_spawn_worker", lambda job_id: spawned.append(job_id))

    result = server.gemma_delegate("do a thing", str(tmp_path))

    assert result["status"] == "running"
    assert spawned == [result["job_id"]]


def test_gemma_delegate_queues_when_at_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db", max_concurrent_jobs=1))
    spawned = []
    monkeypatch.setattr(server, "_spawn_worker", lambda job_id: spawned.append(job_id))

    first = server.gemma_delegate("first", str(tmp_path))
    second = server.gemma_delegate("second", str(tmp_path))

    assert first["status"] == "running"
    assert second["status"] == "queued"
    assert spawned == [first["job_id"]]


def test_gemma_check_returns_status_and_summary_when_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "t", str(tmp_path))
    db.update_status(conn, job_id, "done", result_summary="all finished")
    conn.close()

    result = server.gemma_check(job_id)
    assert result["status"] == "done"
    assert result["result_summary"] == "all finished"


def test_gemma_check_omits_summary_when_running(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "t", str(tmp_path))
    db.update_status(conn, job_id, "running")
    conn.close()

    result = server.gemma_check(job_id)
    assert result["status"] == "running"
    assert "result_summary" not in result


def test_gemma_check_unknown_job_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    db.connect(config.load().db_path).close()
    result = server.gemma_check("nope")
    assert "error" in result


def test_gemma_cancel_sets_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "t", str(tmp_path))
    conn.close()

    result = server.gemma_cancel(job_id)
    assert result["status"] == "cancel_requested"

    conn = db.connect(config.load().db_path)
    assert db.is_cancel_requested(conn, job_id) is True


def test_gemma_cancel_unknown_job_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    db.connect(config.load().db_path).close()
    result = server.gemma_cancel("nope")
    assert "error" in result


def test_gemma_delegate_marks_job_error_when_spawn_fails_after_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db", max_concurrent_jobs=3))

    def _boom(job_id):
        raise OSError("failed to launch interpreter")

    monkeypatch.setattr(server, "_spawn_worker", _boom)

    result = server.gemma_delegate("do a thing", str(tmp_path))

    assert result["status"] == "error"
    job_id = result["job_id"]

    conn = db.connect(config.load().db_path)
    job = db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "error"
    assert "failed to launch interpreter" in job["result_summary"]


def test_gemma_delegate_returns_error_dict_when_db_connect_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))

    def _boom(path):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "connect", _boom)

    result = server.gemma_delegate("do a thing", str(tmp_path))

    assert "error" in result
    assert "job_id" not in result


def test_gemma_check_returns_error_dict_when_db_connect_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))

    def _boom(path):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "connect", _boom)

    result = server.gemma_check("some-id")

    assert "error" in result


def test_gemma_cancel_returns_error_dict_when_db_connect_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))

    def _boom(path):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "connect", _boom)

    result = server.gemma_cancel("some-id")

    assert "error" in result


def test_gemma_check_reports_progress_for_a_running_job(tmp_path, monkeypatch):
    """The whole point of the progress trail: a caller must be able to see
    what a running job is doing without opening its log."""
    from handyman import progress

    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "do a thing", str(tmp_path))
    db.update_status(conn, job_id, "running")
    progress.record(conn, job_id, 1, "chat")
    progress.record(conn, job_id, 2, "tool_call", "write_file")
    conn.close()

    result = server.gemma_check(job_id)

    assert result["status"] == "running"
    assert result["iteration"] == 2
    assert result["last_action"] == "tool_call write_file"
    assert result["recent"] == ["1: chat", "2: tool_call write_file"]


def test_gemma_check_omits_progress_fields_when_nothing_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "do a thing", str(tmp_path))
    conn.close()

    result = server.gemma_check(job_id)

    assert result["status"] == "queued"
    assert "recent" not in result


def test_gemma_check_survives_unreadable_progress_tables(tmp_path, monkeypatch):
    """A status check must never fail because progress lookup broke - the
    job's own status is the thing the caller actually needs."""
    from handyman import progress

    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "do a thing", str(tmp_path))
    conn.close()

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("no such table: job_events")

    monkeypatch.setattr(progress, "heartbeat", boom)
    result = server.gemma_check(job_id)

    assert result["job_id"] == job_id
    assert result["status"] == "queued"


def test_gemma_delegate_records_the_real_worker_pid(tmp_path, monkeypatch):
    """The claim uses a pid=0 placeholder, so the row is briefly
    unattributable. Recording the spawned pid immediately shrinks that
    window to almost nothing instead of relying on the reaper's grace."""
    monkeypatch.setattr(
        config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db")
    )
    monkeypatch.setattr(server, "_spawn_worker", lambda job_id: 4242)

    result = server.gemma_delegate("do a thing", str(tmp_path))

    conn = db.connect(tmp_path / "jobs.db")
    job = db.get_job(conn, result["job_id"])
    conn.close()
    assert job["pid"] == 4242


def test_delegate_records_the_callers_model_and_derives_the_provider(tmp_path, monkeypatch):
    """The caller names a model; where it runs follows from where that
    model lives. Choosing the two independently let them disagree."""
    monkeypatch.setenv("HM_KEY", "k")
    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: make_config(
            tmp_path, db_path=tmp_path / "jobs.db",
            providers={"cloud": {"host": "https://example/openai",
                                 "api_key_env": "HM_KEY"}},
            models=[{"name": "big", "provider": "cloud", "model": "remote-1"}]))
    monkeypatch.setattr(server, "_spawn_worker", lambda job_id: 123)

    result = server.gemma_delegate("do a thing", str(tmp_path), model="big")

    assert result["provider"] == "hosted"
    assert result["model"] == "big"
    conn = db.connect(tmp_path / "jobs.db")
    job = db.get_job(conn, result["job_id"])
    conn.close()
    assert job["provider"] == "hosted"
    assert job["model"] == "big"


def test_delegate_reports_a_hosted_model_with_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("HM_KEY", raising=False)
    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: make_config(
            tmp_path, db_path=tmp_path / "jobs.db",
            providers={"cloud": {"host": "https://example/openai",
                                 "api_key_env": "HM_KEY"}},
            models=[{"name": "big", "provider": "cloud", "model": "remote-1"}]))

    result = server.gemma_delegate("t", str(tmp_path), model="big")

    assert "API key" in result["error"]


def test_delegate_rejects_a_model_the_named_provider_does_not_have(tmp_path, monkeypatch):
    """The whole point of registering a model with its provider: a
    combination that cannot work is refused up front, rather than becoming
    a 404 from whichever endpoint happened to be configured."""
    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))

    result = server.gemma_delegate("t", str(tmp_path), model="definitely-not-real",
                                   provider_name="hosted")

    assert "no model called" in result["error"]


def test_delegate_defaults_to_the_configured_model(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, db_path=tmp_path / "jobs.db")
    monkeypatch.setattr(config, "load", lambda *a, **k: cfg)
    monkeypatch.setattr(server, "_spawn_worker", lambda job_id: 1)
    monkeypatch.setattr(server, "_local_server_reachable", lambda c: True)

    result = server.gemma_delegate("t", str(tmp_path))

    assert result["model"] == cfg.tiers[0].model
    assert result["provider"] == "local"
