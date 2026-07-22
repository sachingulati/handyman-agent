import os
import subprocess
import sys

from handyman import db
def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_create_and_get_job(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "do the thing", "/some/dir")
    job = db.get_job(conn, job_id)
    assert job["id"] == job_id
    assert job["task"] == "do the thing"
    assert job["working_dir"] == "/some/dir"
    assert job["status"] == "queued"
    assert job["cancel_requested"] == 0


def test_get_job_missing_returns_none(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    assert db.get_job(conn, "does-not-exist") is None


def test_count_running(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    j1 = db.create_job(conn, "t1", "/d")
    j2 = db.create_job(conn, "t2", "/d")
    assert db.count_running(conn) == 0
    db.try_claim_with_cap(conn, j1, pid=111, max_concurrent=3)
    assert db.count_running(conn) == 1
    db.try_claim_with_cap(conn, j2, pid=222, max_concurrent=3)
    assert db.count_running(conn) == 2


def test_try_claim_with_cap_succeeds_under_cap(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    assert db.try_claim_with_cap(conn, job_id, pid=1, max_concurrent=3) is True
    assert db.get_job(conn, job_id)["status"] == "running"


def test_try_claim_with_cap_fails_over_cap(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    j1 = db.create_job(conn, "t1", "/d")
    j2 = db.create_job(conn, "t2", "/d")
    db.try_claim_with_cap(conn, j1, pid=1, max_concurrent=1)
    assert db.try_claim_with_cap(conn, j2, pid=2, max_concurrent=1) is False
    assert db.get_job(conn, j2)["status"] == "queued"


def test_try_claim_with_cap_fails_if_not_queued(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.try_claim_with_cap(conn, job_id, pid=1, max_concurrent=3)
    assert db.try_claim_with_cap(conn, job_id, pid=2, max_concurrent=3) is False


def test_claim_next_queued_job_picks_oldest(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    older = db.create_job(conn, "older", "/d")
    conn.execute("UPDATE jobs SET created_at='2020-01-01T00:00:00' WHERE id=?", (older,))
    newer = db.create_job(conn, "newer", "/d")
    conn.execute("UPDATE jobs SET created_at='2030-01-01T00:00:00' WHERE id=?", (newer,))
    conn.commit()

    claimed = db.claim_next_queued_job(conn, pid=99)
    assert claimed == older
    assert db.get_job(conn, older)["status"] == "running"
    assert db.get_job(conn, newer)["status"] == "queued"


def test_claim_next_queued_job_returns_none_when_empty(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    assert db.claim_next_queued_job(conn, pid=1) is None


# --- tier-aware concurrency: prevent Ollama thrashing between models -----


def test_try_claim_with_cap_starts_new_job_on_base_tier(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    assert db.try_claim_with_cap(conn, job_id, pid=1, max_concurrent=3) is True
    assert db.get_job(conn, job_id)["current_tier"] == db.BASE_TIER


def test_try_claim_with_cap_refuses_when_a_running_job_is_on_a_different_tier(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    running = db.create_job(conn, "running", "/d")
    db.try_claim_with_cap(conn, running, pid=1, max_concurrent=3)
    db.set_current_tier(conn, running, "mid")

    new_job = db.create_job(conn, "new", "/d")
    assert db.try_claim_with_cap(conn, new_job, pid=2, max_concurrent=3) is False
    assert db.get_job(conn, new_job)["status"] == "queued"


def test_try_claim_with_cap_refuses_when_a_running_job_is_escalating(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    running = db.create_job(conn, "running", "/d")
    db.try_claim_with_cap(conn, running, pid=1, max_concurrent=3)
    db.set_escalating(conn, running, True)

    new_job = db.create_job(conn, "new", "/d")
    assert db.try_claim_with_cap(conn, new_job, pid=2, max_concurrent=3) is False
    assert db.get_job(conn, new_job)["status"] == "queued"


def test_try_claim_with_cap_succeeds_when_running_jobs_share_base_tier(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    running = db.create_job(conn, "running", "/d")
    db.try_claim_with_cap(conn, running, pid=1, max_concurrent=3)

    new_job = db.create_job(conn, "new", "/d")
    assert db.try_claim_with_cap(conn, new_job, pid=2, max_concurrent=3) is True


def test_claim_next_queued_job_refuses_when_a_running_job_is_on_a_different_tier(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    running = db.create_job(conn, "running", "/d")
    db.try_claim_with_cap(conn, running, pid=1, max_concurrent=3)
    db.set_current_tier(conn, running, "big")

    queued = db.create_job(conn, "queued", "/d")
    assert db.claim_next_queued_job(conn, pid=2) is None
    assert db.get_job(conn, queued)["status"] == "queued"


def test_claim_next_queued_job_refuses_when_a_running_job_is_escalating(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    running = db.create_job(conn, "running", "/d")
    db.try_claim_with_cap(conn, running, pid=1, max_concurrent=3)
    db.set_escalating(conn, running, True)

    db.create_job(conn, "queued", "/d")
    assert db.claim_next_queued_job(conn, pid=2) is None


def test_set_current_tier_updates_tier(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.set_current_tier(conn, job_id, "big")
    assert db.get_job(conn, job_id)["current_tier"] == "big"


def test_set_escalating_updates_flag(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.set_escalating(conn, job_id, True)
    assert db.get_job(conn, job_id)["escalating"] == 1
    db.set_escalating(conn, job_id, False)
    assert db.get_job(conn, job_id)["escalating"] == 0


def test_is_sole_runner_true_when_no_other_job_running(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.try_claim_with_cap(conn, job_id, pid=1, max_concurrent=3)
    assert db.is_sole_runner(conn, job_id) is True


def test_is_sole_runner_false_when_another_job_is_running(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    j1 = db.create_job(conn, "t1", "/d")
    j2 = db.create_job(conn, "t2", "/d")
    db.try_claim_with_cap(conn, j1, pid=1, max_concurrent=3)
    db.try_claim_with_cap(conn, j2, pid=2, max_concurrent=3)
    assert db.is_sole_runner(conn, j1) is False


def test_update_status_sets_result_and_transcript(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.update_status(conn, job_id, "done", result_summary="all good", transcript_path="/log.txt")
    job = db.get_job(conn, job_id)
    assert job["status"] == "done"
    assert job["result_summary"] == "all good"
    assert job["transcript_path"] == "/log.txt"


def test_update_status_preserves_existing_fields_when_none(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.update_status(conn, job_id, "running", transcript_path="/log.txt")
    db.update_status(conn, job_id, "done", result_summary="finished")
    job = db.get_job(conn, job_id)
    assert job["transcript_path"] == "/log.txt"
    assert job["result_summary"] == "finished"


def test_touch_updates_timestamp(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    before = db.get_job(conn, job_id)["updated_at"]
    conn.execute("UPDATE jobs SET updated_at='2000-01-01T00:00:00' WHERE id=?", (job_id,))
    conn.commit()
    db.touch(conn, job_id)
    after = db.get_job(conn, job_id)["updated_at"]
    assert after != "2000-01-01T00:00:00"
    assert after >= before


def test_request_cancel_and_is_cancel_requested(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    assert db.is_cancel_requested(conn, job_id) is False
    assert db.request_cancel(conn, job_id) is True
    assert db.is_cancel_requested(conn, job_id) is True


def test_request_cancel_unknown_job_returns_false(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    assert db.request_cancel(conn, "nope") is False


def test_is_pid_alive_true_for_current_process():
    assert db.is_pid_alive(os.getpid()) is True


def test_is_pid_alive_false_for_exited_process():
    assert db.is_pid_alive(_dead_pid()) is False


def test_is_pid_alive_false_for_zero():
    assert db.is_pid_alive(0) is False


def test_reap_dead_running_jobs_marks_dead_pid_as_error(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.try_claim_with_cap(conn, job_id, pid=_dead_pid(), max_concurrent=3)

    db.reap_dead_running_jobs(conn)

    job = db.get_job(conn, job_id)
    assert job["status"] == "error"
    assert "no longer running" in job["result_summary"]


def test_reap_dead_running_jobs_leaves_alive_pid_running(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.try_claim_with_cap(conn, job_id, pid=os.getpid(), max_concurrent=3)

    db.reap_dead_running_jobs(conn)

    assert db.get_job(conn, job_id)["status"] == "running"


def test_reap_dead_running_jobs_ignores_not_yet_started_pid(tmp_path):
    # pid=0 means "claimed but the real worker hasn't called set_pid yet" -
    # a legitimate transient state, not a dead process, and must not be reaped.
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    db.try_claim_with_cap(conn, job_id, pid=0, max_concurrent=3)

    db.reap_dead_running_jobs(conn)

    assert db.get_job(conn, job_id)["status"] == "running"


def test_reap_dead_running_jobs_frees_a_wedged_concurrency_slot(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    stuck_job = db.create_job(conn, "stuck", "/d")
    db.try_claim_with_cap(conn, stuck_job, pid=_dead_pid(), max_concurrent=1)
    assert db.count_running(conn) == 1

    new_job = db.create_job(conn, "new", "/d")
    assert db.try_claim_with_cap(conn, new_job, pid=os.getpid(), max_concurrent=1) is False

    db.reap_dead_running_jobs(conn)

    assert db.try_claim_with_cap(conn, new_job, pid=os.getpid(), max_concurrent=1) is True


def test_connect_creates_a_missing_data_directory(tmp_path):
    """On a fresh install the data dir does not exist yet, and sqlite3
    reports only 'unable to open database file' - the first thing a new
    user would ever see."""
    nested = tmp_path / "does" / "not" / "exist" / "jobs.db"
    conn = db.connect(nested)
    conn.close()
    assert nested.exists()
