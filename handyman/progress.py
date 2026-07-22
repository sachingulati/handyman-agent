"""Records job progress so a caller can check status without reading logs."""
import datetime


def ensure_schema(conn) -> None:
    """Creates the job_events table and updates the jobs table columns."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            iteration INTEGER,
            event_type TEXT NOT NULL,
            detail TEXT
        )
        """
    )

    # CREATE TABLE IF NOT EXISTS is a no-op on an existing jobs table, so
    # databases created before these columns existed need them added
    # explicitly. Same idempotent-migration pattern as db.connect().
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "iteration" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN iteration INTEGER DEFAULT 0")
    if "last_action" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN last_action TEXT")


def record(conn, job_id: str, iteration: int, event_type: str, detail=None) -> None:
    """Inserts a new row into job_events and updates the heartbeat fields for that job."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    last_action = f"{event_type} {detail}" if detail is not None else event_type

    conn.execute(
        "INSERT INTO job_events (job_id, ts, iteration, event_type, detail) VALUES (?, ?, ?, ?, ?)",
        (job_id, ts, iteration, event_type, detail),
    )
    conn.execute(
        "UPDATE jobs SET iteration = ?, last_action = ? WHERE id = ?",
        (iteration, last_action, job_id),
    )
    conn.commit()


def recent_events(conn, job_id: str, limit: int = 10) -> list[dict]:
    """Returns a list of the most recent events for a specific job."""
    # DESC + LIMIT takes the newest rows; reversing restores oldest-first
    # for the caller. Ordering by id, not ts, so events recorded inside the
    # same clock tick keep their insertion order.
    query = "SELECT iteration, event_type, detail, ts FROM job_events WHERE job_id = ? ORDER BY id DESC LIMIT ?"
    rows = list(reversed(conn.execute(query, (job_id, limit)).fetchall()))
    return [
        {"iteration": r[0], "event_type": r[1], "detail": r[2], "ts": r[3]}
        for r in rows
    ]


def heartbeat(conn, job_id: str) -> dict | None:
    """Returns a dictionary of the latest iteration and last_action for a given job."""
    row = conn.execute("SELECT iteration, last_action FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return {"iteration": row[0], "last_action": row[1]}
