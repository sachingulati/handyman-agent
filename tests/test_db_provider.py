from handyman import db


def test_provider_defaults_to_local(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    assert db.get_job(conn, job_id)["provider"] == "local"


def test_create_job_records_the_provider(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d", provider="hosted")
    assert db.get_job(conn, job_id)["provider"] == "hosted"


def test_hosted_jobs_do_not_count_against_the_local_cap(tmp_path):
    """The concurrency cap exists because one GPU holds one model at a
    time. A hosted job occupies no VRAM, so counting it would block local
    work for no reason."""
    conn = db.connect(tmp_path / "jobs.db")
    for _ in range(3):
        hosted = db.create_job(conn, "t", "/d", provider="hosted")
        db.try_claim_with_cap(conn, hosted, pid=1, max_concurrent=1, provider="hosted")
    local = db.create_job(conn, "t", "/d")
    assert db.try_claim_with_cap(conn, local, pid=2, max_concurrent=1) is True


def test_local_jobs_still_respect_the_cap(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    first = db.create_job(conn, "t", "/d")
    db.try_claim_with_cap(conn, first, pid=1, max_concurrent=1)
    second = db.create_job(conn, "t", "/d")
    assert db.try_claim_with_cap(conn, second, pid=2, max_concurrent=1) is False


def test_hosted_claim_is_never_refused(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    for _ in range(5):
        db.try_claim_with_cap(conn, db.create_job(conn, "t", "/d"), pid=1, max_concurrent=1)
    hosted = db.create_job(conn, "t", "/d", provider="hosted")
    assert db.try_claim_with_cap(conn, hosted, pid=9, max_concurrent=1,
                                 provider="hosted") is True


def test_count_running_local_ignores_hosted(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    h = db.create_job(conn, "t", "/d", provider="hosted")
    db.try_claim_with_cap(conn, h, pid=1, max_concurrent=5, provider="hosted")
    assert db.count_running(conn) == 1
    assert db.count_running_local(conn) == 0


def test_existing_database_gains_the_provider_column(tmp_path):
    """An idempotent migration, as with every other column added here."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, task TEXT NOT NULL, "
                "working_dir TEXT NOT NULL, status TEXT NOT NULL, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    old.commit()
    old.close()

    conn = db.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert "provider" in cols


def test_model_defaults_to_empty(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d")
    assert db.get_job(conn, job_id)["model"] == ""


def test_create_job_records_the_requested_model(tmp_path):
    """The caller chooses the model, so the choice has to survive the hop
    into the worker process."""
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "t", "/d", model="qwen3:14b")
    assert db.get_job(conn, job_id)["model"] == "qwen3:14b"


def test_existing_database_gains_the_model_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, task TEXT NOT NULL, "
                "working_dir TEXT NOT NULL, status TEXT NOT NULL, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    old.commit()
    old.close()
    conn = db.connect(path)
    assert "model" in {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
