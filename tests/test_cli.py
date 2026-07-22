from handyman import cli
from conftest import make_config
from handyman import config
from handyman import db
def test_cli_delegate_prints_job_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    monkeypatch.setattr("handyman.server._spawn_worker", lambda job_id: None)

    code = cli.main(["delegate", "do a thing", str(tmp_path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "job_id" in out
    assert "running" in out


def test_cli_delegate_missing_working_dir_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    code = cli.main(["delegate", "do a thing", str(tmp_path / "missing")])
    assert code == 1
    assert "error" in capsys.readouterr().out


def test_cli_check_prints_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "t", str(tmp_path))
    db.update_status(conn, job_id, "done", result_summary="finished")
    conn.close()

    code = cli.main(["check", job_id])

    assert code == 0
    out = capsys.readouterr().out
    assert "done" in out
    assert "finished" in out


def test_cli_cancel_prints_confirmation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "load", lambda *a, **k: make_config(tmp_path, db_path=tmp_path / "jobs.db"))
    conn = db.connect(config.load().db_path)
    job_id = db.create_job(conn, "t", str(tmp_path))
    conn.close()

    code = cli.main(["cancel", job_id])

    assert code == 0
    assert "cancel_requested" in capsys.readouterr().out


def test_cli_unknown_command_returns_error(capsys):
    code = cli.main(["bogus"])
    assert code == 1
    assert "unknown command" in capsys.readouterr().out
