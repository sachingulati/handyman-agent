import sqlite3
from pathlib import Path
import uuid
from datetime import datetime, timezone

from handyman.procutil import is_pid_alive  # re-exported: db.is_pid_alive is a documented call site


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def reap_dead_running_jobs(conn: sqlite3.Connection) -> None:
    """Mark 'running' jobs whose worker process is no longer alive as 'error'.

    A worker can disappear without ever updating its own job row (killed
    externally, crashed before its try/except, or hung during model pull).
    Left unreaped, such a job stays 'running' forever and permanently
    occupies one of the concurrency slots, eventually wedging the whole
    queue. Call this before any capacity check or status read.
    """
    now = datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT id, pid, updated_at FROM jobs WHERE status='running'"
    ).fetchall()
    for job_id, pid, updated_at in rows:
        if pid:
            stranded = not is_pid_alive(pid)
            reason = "worker process is no longer running"
        else:
            # No pid recorded. That is legitimate for a moment after the
            # claim and before the worker registers itself, so only treat
            # it as stranded once it has been that way for a while.
            # Reaping on a falsy pid alone would kill every job the
            # instant it started.
            try:
                age = (now - datetime.fromisoformat(updated_at)).total_seconds()
            except (TypeError, ValueError):
                age = 0
            stranded = age > UNSET_PID_GRACE_SECONDS
            reason = "worker never started (no pid recorded)"

        if stranded:
            conn.execute(
                "UPDATE jobs SET status='error', "
                "result_summary=COALESCE(result_summary, ?), "
                "updated_at=? WHERE id=? AND status='running'",
                (reason, now_iso(), job_id),
            )
    conn.commit()


BASE_TIER = "small"

# How long a 'running' job may sit with no pid recorded before the reaper
# treats it as stranded. A job is claimed before its worker exists, so
# there is always a brief, legitimate window where the pid is unknown;
# this must be comfortably longer than that window and shorter than a
# user's patience.
UNSET_PID_GRACE_SECONDS = 120


def connect(db_path) -> sqlite3.Connection:
    # The data directory may not exist yet on a fresh install; sqlite3
    # reports a bare "unable to open database file" if it is missing,
    # which is the first thing a new user would ever see.
    parent = Path(db_path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            working_dir TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            result_summary TEXT,
            transcript_path TEXT,
            pid INTEGER,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            current_tier TEXT NOT NULL DEFAULT 'small',
            escalating INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Idempotent migration for jobs.db files created before these columns
    # existed - CREATE TABLE IF NOT EXISTS above is a no-op on an existing
    # table, so older databases need the columns added explicitly.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "current_tier" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN current_tier TEXT NOT NULL DEFAULT 'small'")
    if "escalating" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN escalating INTEGER NOT NULL DEFAULT 0")
    conn.commit()

    from handyman import progress  # imported here to avoid a circular import at module load
    progress.ensure_schema(conn)
    conn.commit()
    return conn


def create_job(conn: sqlite3.Connection, task: str, working_dir: str) -> str:
    job_id = uuid.uuid4().hex
    ts = now_iso()
    conn.execute(
        "INSERT INTO jobs (id, task, working_dir, status, created_at, updated_at, cancel_requested) "
        "VALUES (?, ?, ?, 'queued', ?, ?, 0)",
        (job_id, task, working_dir, ts, ts),
    )
    conn.commit()
    return job_id


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, task, working_dir, status, created_at, updated_at, "
        "result_summary, transcript_path, pid, cancel_requested, "
        "current_tier, escalating "
        "FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    keys = [
        "id", "task", "working_dir", "status", "created_at", "updated_at",
        "result_summary", "transcript_path", "pid", "cancel_requested",
        "current_tier", "escalating",
    ]
    return dict(zip(keys, row))


def count_running(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()
    return row[0]


def try_claim_with_cap(conn: sqlite3.Connection, job_id: str, pid: int, max_concurrent: int) -> bool:
    # A freshly claimed job always starts on BASE_TIER, so it may only join
    # the running cohort if no running job is on a different tier and none
    # is mid-escalation - otherwise Ollama would thrash between two loaded
    # models (measured live: ~9s reload on every alternating request).
    cur = conn.execute(
        """
        UPDATE jobs SET status='running', pid=?, updated_at=?, current_tier=?
        WHERE id=? AND status='queued'
          AND (SELECT COUNT(*) FROM jobs WHERE status='running') < ?
          AND (SELECT COUNT(*) FROM jobs WHERE status='running' AND current_tier != ?) = 0
          AND (SELECT COUNT(*) FROM jobs WHERE status='running' AND escalating != 0) = 0
        """,
        (pid, now_iso(), BASE_TIER, job_id, max_concurrent, BASE_TIER),
    )
    conn.commit()
    return cur.rowcount == 1


def claim_next_queued_job(conn: sqlite3.Connection, pid: int) -> str | None:
    cur = conn.execute(
        """
        UPDATE jobs SET status='running', pid=?, updated_at=?, current_tier=?
        WHERE id = (
            SELECT id FROM jobs WHERE status='queued'
              AND (SELECT COUNT(*) FROM jobs WHERE status='running' AND current_tier != ?) = 0
              AND (SELECT COUNT(*) FROM jobs WHERE status='running' AND escalating != 0) = 0
            ORDER BY created_at LIMIT 1
        )
          AND status='queued'
        RETURNING id
        """,
        (pid, now_iso(), BASE_TIER, BASE_TIER),
    )
    row = cur.fetchone()
    conn.commit()
    return row[0] if row else None


def set_current_tier(conn: sqlite3.Connection, job_id: str, tier: str) -> None:
    conn.execute(
        "UPDATE jobs SET current_tier=?, updated_at=? WHERE id=?",
        (tier, now_iso(), job_id),
    )
    conn.commit()


def set_escalating(conn: sqlite3.Connection, job_id: str, escalating: bool) -> None:
    conn.execute(
        "UPDATE jobs SET escalating=?, updated_at=? WHERE id=?",
        (1 if escalating else 0, now_iso(), job_id),
    )
    conn.commit()


def is_sole_runner(conn: sqlite3.Connection, job_id: str) -> bool:
    """True if no other job is currently 'running' - i.e. it's safe for this
    job to switch model tiers without another job thrashing against it."""
    row = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status='running' AND id != ?", (job_id,)
    ).fetchone()
    return row[0] == 0


def set_pid(conn: sqlite3.Connection, job_id: str, pid: int) -> None:
    conn.execute("UPDATE jobs SET pid=?, updated_at=? WHERE id=?", (pid, now_iso(), job_id))
    conn.commit()


def update_status(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    result_summary: str | None = None,
    transcript_path: str | None = None,
) -> None:
    conn.execute(
        "UPDATE jobs SET status=?, "
        "result_summary=COALESCE(?, result_summary), "
        "transcript_path=COALESCE(?, transcript_path), "
        "updated_at=? WHERE id=?",
        (status, result_summary, transcript_path, now_iso(), job_id),
    )
    conn.commit()


def touch(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute("UPDATE jobs SET updated_at=? WHERE id=?", (now_iso(), job_id))
    conn.commit()


def request_cancel(conn: sqlite3.Connection, job_id: str) -> bool:
    cur = conn.execute(
        "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?",
        (now_iso(), job_id),
    )
    conn.commit()
    return cur.rowcount == 1


def is_cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
    return bool(row[0]) if row else False
