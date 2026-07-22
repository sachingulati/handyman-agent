from pathlib import Path

from mcp.server.fastmcp import FastMCP

import config
import db
import worker

mcp = FastMCP("gemma-agent")


def _spawn_worker(job_id: str) -> None:
    worker.spawn_worker(job_id)


@mcp.tool()
def gemma_delegate(task: str, working_dir: str) -> dict:
    """Delegate a task to a local Gemma 4 subagent that runs in the background.

    Use this for mechanical/low-stakes work: bulk file edits, research and
    summarization, draft generation, or long-running background tasks that
    don't need Claude's own reasoning. Returns immediately with a job_id —
    call gemma_check(job_id) later to see progress or get the result.
    """
    if not Path(working_dir).is_dir():
        return {"error": f"working_dir does not exist: {working_dir}"}

    conn = None
    try:
        conn = db.connect(config.DB_PATH)
        db.reap_dead_running_jobs(conn)
        job_id = db.create_job(conn, task, working_dir)
        claimed = db.try_claim_with_cap(
            conn, job_id, pid=0, max_concurrent=config.MAX_CONCURRENT_JOBS
        )

        if claimed:
            try:
                _spawn_worker(job_id)
            except Exception as exc:
                db.update_status(
                    conn, job_id, "error", result_summary=f"failed to spawn worker: {exc}"
                )
                return {"job_id": job_id, "status": "error"}
            return {"job_id": job_id, "status": "running"}
        return {"job_id": job_id, "status": "queued"}
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if conn is not None:
            conn.close()


@mcp.tool()
def gemma_check(job_id: str) -> dict:
    """Check the status of a job started with gemma_delegate."""
    conn = None
    try:
        conn = db.connect(config.DB_PATH)
        db.reap_dead_running_jobs(conn)
        job = db.get_job(conn, job_id)
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if conn is not None:
            conn.close()

    if job is None:
        return {"error": f"no such job: {job_id}"}

    response = {"job_id": job["id"], "status": job["status"]}
    if job["status"] in ("done", "incomplete", "timeout", "error", "canceled"):
        response["result_summary"] = job["result_summary"]
    return response


@mcp.tool()
def gemma_cancel(job_id: str) -> dict:
    """Cancel a running or queued job started with gemma_delegate."""
    conn = None
    try:
        conn = db.connect(config.DB_PATH)
        ok = db.request_cancel(conn, job_id)
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if conn is not None:
            conn.close()

    if not ok:
        return {"error": f"no such job: {job_id}"}
    return {"job_id": job_id, "status": "cancel_requested"}


if __name__ == "__main__":
    db.connect(config.DB_PATH).close()  # ensure schema exists at startup
    mcp.run()
