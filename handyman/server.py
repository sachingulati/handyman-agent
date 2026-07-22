from pathlib import Path

from mcp.server.fastmcp import FastMCP

from handyman import config
from handyman import db
from handyman import progress
from handyman import provider
from handyman import worker

mcp = FastMCP("gemma-agent")


def _local_server_reachable(cfg) -> bool:
    """Whether the local model server answers.

    Checked before choosing a provider rather than after a job fails, so
    an outage produces an actionable message instead of a traceback in a
    result summary.
    """
    if not cfg.tiers:
        return False
    try:
        import requests

        requests.get(f"{cfg.ollama_host}/api/tags", timeout=5).raise_for_status()
        return True
    except Exception:
        return False


def _spawn_worker(job_id: str) -> int:
    return worker.spawn_worker(job_id)


@mcp.tool()
def gemma_delegate(task: str, working_dir: str, allow_hosted: bool = False) -> dict:
    """Delegate a task to a local Gemma 4 subagent that runs in the background.

    Use this for mechanical/low-stakes work: bulk file edits, research and
    summarization, draft generation, or long-running background tasks that
    don't need Claude's own reasoning. Returns immediately with a job_id —
    call gemma_check(job_id) later to see progress or get the result.

    allow_hosted lets the job run on a hosted model when the local one is
    busy or unreachable. It is off by default: hosted work leaves this
    machine, so it is never chosen for you.
    """
    cfg = config.load()
    if not Path(working_dir).is_dir():
        return {"error": f"working_dir does not exist: {working_dir}"}

    conn = None
    try:
        conn = db.connect(cfg.db_path)
        db.reap_dead_running_jobs(conn)

        at_capacity = db.count_running_local(conn) >= cfg.max_concurrent_jobs
        try:
            chosen = provider.choose(
                cfg,
                local_available=_local_server_reachable(cfg),
                at_capacity=at_capacity,
                allow_hosted=allow_hosted,
            )
        except provider.ProviderUnavailable as exc:
            return {"error": str(exc)}

        job_id = db.create_job(conn, task, working_dir, provider=chosen)
        claimed = db.try_claim_with_cap(
            conn, job_id, pid=0, max_concurrent=cfg.max_concurrent_jobs,
            provider=chosen,
        )

        if claimed:
            try:
                # Record the real pid as soon as it exists. The claim above
                # used a placeholder, leaving the row briefly unattributable
                # to any process; closing that window fast means the reaper's
                # grace period is a backstop rather than the primary defence.
                worker_pid = _spawn_worker(job_id)
                if worker_pid:
                    db.set_pid(conn, job_id, worker_pid)
            except Exception as exc:
                db.update_status(
                    conn, job_id, "error", result_summary=f"failed to spawn worker: {exc}"
                )
                return {"job_id": job_id, "status": "error"}
            return {"job_id": job_id, "status": "running", "provider": chosen}
        return {"job_id": job_id, "status": "queued", "provider": chosen}
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if conn is not None:
            conn.close()


@mcp.tool()
def gemma_check(job_id: str) -> dict:
    """Check the status of a job started with gemma_delegate."""
    cfg = config.load()
    conn = None
    try:
        conn = db.connect(cfg.db_path)
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

    # Progress detail so a caller can see what a running job is doing
    # without reading its log. Best-effort: a status check must never fail
    # because the progress tables are missing or unreadable.
    conn = None
    try:
        conn = db.connect(cfg.db_path)
        hb = progress.heartbeat(conn, job_id)
        if hb:
            response["iteration"] = hb["iteration"]
            response["last_action"] = hb["last_action"]
        events = progress.recent_events(conn, job_id, limit=5)
        if events:
            response["recent"] = [
                f"{e['iteration']}: {e['event_type']}"
                + (f" {e['detail']}" if e["detail"] else "")
                for e in events
            ]
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()
    return response


@mcp.tool()
def gemma_cancel(job_id: str) -> dict:
    """Cancel a running or queued job started with gemma_delegate."""
    cfg = config.load()
    conn = None
    try:
        conn = db.connect(cfg.db_path)
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
    db.connect(config.load().db_path).close()  # ensure schema exists at startup
    mcp.run()
