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


def test_cli_unknown_command_exits_with_usage(capsys):
    # argparse rejects an unknown subcommand itself, and its message lists
    # the real ones - more useful than a bare error code.
    import pytest

    with pytest.raises(SystemExit) as exc:
        cli.main(["bogus"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_exposes_the_operational_commands():
    """The machinery around delegation - waiting, batching, checking a
    workspace, finding orphaned workers - belongs in the tool. Rebuilding
    it around every use is where most of the cost went."""
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    names = set(actions[0].choices)
    assert {"delegate", "check", "cancel"} <= names, "job verbs"
    assert {"run", "batch"} <= names, "run to completion, and queues"
    assert {"doctor", "ps", "reset"} <= names, "keeping results trustworthy"


def test_run_accepts_a_task_file_or_literal_text(tmp_path):
    brief = tmp_path / "brief.md"
    brief.write_text("Write a file, then stop.", encoding="utf-8")
    args = cli.build_parser().parse_args(["run", str(brief), str(tmp_path)])
    assert args.task == str(brief)
    args = cli.build_parser().parse_args(["run", "inline task text", str(tmp_path)])
    assert args.task == "inline task text"


def test_models_command_runs_and_separates_enabled_from_offered(tmp_path, monkeypatch, capsys):
    """This command had a NameError that the whole suite missed, because
    nothing exercised it. Cheap to cover, and it is the surface a
    delegating model reads before choosing."""
    from conftest import make_config
    from handyman import registry

    monkeypatch.setenv("HM_KEY", "k")
    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: make_config(
            tmp_path,
            providers={"local": {"host": "http://x"},
                       "cloud": {"host": "http://y", "api_key_env": "HM_KEY"}},
            models=[{"name": "big", "provider": "cloud", "model": "m", "cost": "free"}]))
    monkeypatch.setattr(registry, "discover",
                        lambda p, timeout=15: ["free-local"] if not p.hosted else ["pricey"])

    args = cli.build_parser().parse_args(["models"])
    assert args.func(args) == 0

    out = capsys.readouterr().out
    assert "enabled" in out and "available, not enabled" in out
    assert "big" in out and "free-local" in out
    assert "pricey" in out
