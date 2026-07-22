import sqlite3

import pytest

from handyman import db, progress


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "jobs.db")
    progress.ensure_schema(c)
    yield c
    c.close()


def _job(c):
    return db.create_job(c, "task", "/tmp")


def test_ensure_schema_is_idempotent(conn):
    """Called on every connect, so it must tolerate already-existing schema."""
    progress.ensure_schema(conn)
    progress.ensure_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "job_events" in tables


def test_heartbeat_columns_exist_on_jobs(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert "iteration" in cols
    assert "last_action" in cols


def test_record_appends_an_event(conn):
    job_id = _job(conn)
    progress.record(conn, job_id, 1, "tool_call", "write_file a.txt")
    events = progress.recent_events(conn, job_id)
    assert len(events) == 1
    assert events[0]["iteration"] == 1
    assert events[0]["event_type"] == "tool_call"
    assert events[0]["detail"] == "write_file a.txt"
    assert events[0]["ts"]


def test_record_updates_the_heartbeat_in_the_same_call(conn):
    """The heartbeat and the event log must never disagree - they are
    written together so a reader cannot observe one without the other."""
    job_id = _job(conn)
    progress.record(conn, job_id, 3, "tool_call", "edit_file cli.py")
    hb = progress.heartbeat(conn, job_id)
    assert hb["iteration"] == 3
    assert hb["last_action"] == "tool_call edit_file cli.py"


def test_record_without_detail_uses_event_type_alone(conn):
    job_id = _job(conn)
    progress.record(conn, job_id, 2, "chat")
    hb = progress.heartbeat(conn, job_id)
    assert hb["last_action"] == "chat"
    assert progress.recent_events(conn, job_id)[0]["detail"] is None


def test_recent_events_returns_newest_last(conn):
    job_id = _job(conn)
    for i in range(1, 4):
        progress.record(conn, job_id, i, "tool_call", f"step{i}")
    details = [e["detail"] for e in progress.recent_events(conn, job_id)]
    assert details == ["step1", "step2", "step3"]


def test_recent_events_respects_limit_keeping_the_newest(conn):
    job_id = _job(conn)
    for i in range(1, 11):
        progress.record(conn, job_id, i, "tool_call", f"step{i}")
    details = [e["detail"] for e in progress.recent_events(conn, job_id, limit=3)]
    assert details == ["step8", "step9", "step10"]


def test_events_are_scoped_to_their_job(conn):
    a, b = _job(conn), _job(conn)
    progress.record(conn, a, 1, "tool_call", "for-a")
    progress.record(conn, b, 1, "tool_call", "for-b")
    assert [e["detail"] for e in progress.recent_events(conn, a)] == ["for-a"]
    assert [e["detail"] for e in progress.recent_events(conn, b)] == ["for-b"]


def test_heartbeat_for_unknown_job_is_none(conn):
    assert progress.heartbeat(conn, "does-not-exist") is None


def test_recent_events_for_unknown_job_is_empty(conn):
    assert progress.recent_events(conn, "does-not-exist") == []


def test_heartbeat_defaults_before_any_record(conn):
    job_id = _job(conn)
    hb = progress.heartbeat(conn, job_id)
    assert hb["iteration"] == 0
    assert hb["last_action"] is None
