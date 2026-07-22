import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from handyman import config
from handyman import db
from handyman import ollama_client
from handyman import procutil
from handyman import progress
from handyman import tools

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file relative to the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a text file relative to the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact occurrence of old_str with new_str in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command inside the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its extracted text content.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a list of {url, title} results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a background work agent. You will be given a task and must "
    "complete it fully using the available tools before stopping. "
    "When the task is completely finished, respond with plain text that "
    "includes the exact marker TASK_COMPLETE, followed by a one-paragraph "
    "summary of what you did. Do not write TASK_COMPLETE until the task is "
    "actually done."
)

WATCHDOG_NUDGE = (
    "You have not finished the task and did not call a tool. If you are "
    "actually done, respond again with the exact marker TASK_COMPLETE "
    "followed by your summary. Otherwise, continue the task now by "
    "calling the appropriate tool."
)


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap ~4-chars-per-token estimate, good enough for a growth threshold."""
    return len(json.dumps(messages)) // 4


ESCALATION_POLL_INTERVAL_SECONDS = 2


def run_job(
    conn,
    job_id: str,
    task: str,
    working_dir: str,
    log_path,
    chat_fn,
    max_iterations: int,
    max_wall_clock_seconds: float,
    watchdog_max_retries: int,
    time_fn=time.monotonic,
    execute_tool_fn=None,
    escalation_tiers=None,
    sleep_fn=time.sleep,
) -> None:
    """escalation_tiers: optional list of (threshold_tokens, tier_name,
    chat_fn) triples, sorted ascending by threshold. Each iteration uses
    the chat_fn of the highest tier whose threshold the estimated
    conversation size has reached, falling back to the base chat_fn below
    all thresholds.

    Only one model tier can be GPU-resident at a time on this hardware, so
    switching tiers while a sibling job is running on a different tier
    causes Ollama to thrash (measured live: ~9s reload per alternating
    request). Before actually switching, a job marks itself 'escalating'
    (which blocks new claims - see db.try_claim_with_cap/claim_next_queued_job)
    and waits until it is the only running job, so the switch never
    collides with a sibling still using the old tier."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    start_time = time_fn()
    watchdog_retries = 0
    escalation_tiers = escalation_tiers or []
    current_tier = db.BASE_TIER

    with open(log_path, "a", encoding="utf-8") as log:
        for iteration in range(1, max_iterations + 1):
            if db.is_cancel_requested(conn, job_id):
                db.update_status(conn, job_id, "canceled", result_summary="canceled by user")
                log.write("\n[canceled by user]\n")
                return

            if time_fn() - start_time > max_wall_clock_seconds:
                db.update_status(
                    conn, job_id, "timeout",
                    result_summary="exceeded max wall-clock time",
                )
                log.write("\n[timeout: max wall-clock time exceeded]\n")
                return

            active_chat_fn = chat_fn
            desired_tier = db.BASE_TIER
            tokens = estimate_tokens(messages)
            for threshold, tier_name, tier_fn in escalation_tiers:
                if tokens >= threshold:
                    desired_tier = tier_name
                    active_chat_fn = tier_fn

            if desired_tier != current_tier:
                db.set_escalating(conn, job_id, True)
                while not db.is_sole_runner(conn, job_id):
                    if db.is_cancel_requested(conn, job_id):
                        db.set_escalating(conn, job_id, False)
                        db.update_status(conn, job_id, "canceled", result_summary="canceled by user")
                        log.write("\n[canceled by user while waiting to escalate]\n")
                        return
                    if time_fn() - start_time > max_wall_clock_seconds:
                        db.set_escalating(conn, job_id, False)
                        db.update_status(
                            conn, job_id, "timeout",
                            result_summary="exceeded max wall-clock time while waiting to escalate",
                        )
                        log.write("\n[timeout: max wall-clock time exceeded while waiting to escalate]\n")
                        return
                    log.write(
                        f"\n[waiting to escalate to {desired_tier}: other jobs "
                        f"still running on {current_tier}]\n"
                    )
                    sleep_fn(ESCALATION_POLL_INTERVAL_SECONDS)
                db.set_current_tier(conn, job_id, desired_tier)
                progress.record(conn, job_id, iteration, "escalate", desired_tier)
                db.set_escalating(conn, job_id, False)
                log.write(
                    f"\n[escalated to {desired_tier}: conversation ~{tokens} "
                    f"tokens]\n"
                )
                current_tier = desired_tier

            # chat_fn call stays OUTSIDE the guard below: in the real (non-test)
            # wiring it calls into ollama_client.chat(), which raises
            # OllamaError for genuine backend failures (e.g. "Ollama isn't
            # running", "model not pulled"). Those must propagate out of
            # run_job entirely so worker.main() can mark the job "error"
            # instead of "incomplete". Everything that processes chat_fn's
            # *return value*, however unreliable it may be, is guarded.
            message = active_chat_fn(messages)

            try:
                messages.append(message)
                log.write(f"\n--- iteration {iteration} ---\n{json.dumps(message)}\n")
                db.touch(conn, job_id)
                progress.record(conn, job_id, iteration, "chat")

                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    watchdog_retries = 0
                    for call in tool_calls:
                        fn = call["function"]
                        name = fn["name"]
                        arguments = json.loads(fn["arguments"])
                        progress.record(conn, job_id, iteration, "tool_call", name)
                        result = execute_tool_fn(name, arguments)
                        log.write(f"[tool {name}] -> {result}\n")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": result,
                            }
                        )
                    continue

                content = message.get("content") or ""
                if "TASK_COMPLETE" in content:
                    summary = content.replace("TASK_COMPLETE", "").strip()
                    progress.record(conn, job_id, iteration, "done")
                    db.update_status(conn, job_id, "done", result_summary=summary)
                    log.write("\n[done]\n")
                    return

                watchdog_retries += 1
                progress.record(conn, job_id, iteration, "nudge", f"attempt {watchdog_retries}")
                if watchdog_retries > watchdog_max_retries:
                    db.update_status(
                        conn, job_id, "incomplete",
                        result_summary=(
                            f"stopped without finishing after {watchdog_retries} "
                            f"nudges: {content}"
                        ),
                    )
                    log.write("\n[incomplete: watchdog retries exhausted]\n")
                    return
                messages.append({"role": "user", "content": WATCHDOG_NUDGE})
            except Exception as exc:
                db.update_status(
                    conn, job_id, "incomplete",
                    result_summary=f"tool dispatch failed: {exc}",
                )
                log.write(f"\n[incomplete: tool dispatch failed: {exc}]\n")
                return

        db.update_status(conn, job_id, "timeout", result_summary="exceeded max iterations")
        log.write("\n[timeout: max iterations exceeded]\n")


def execute_tool_call(working_dir: str, name: str, arguments: dict, cfg) -> str:
    try:
        if name == "read_file":
            return tools.read_file(working_dir, arguments["path"])
        if name == "write_file":
            return tools.write_file(working_dir, arguments["path"], arguments["content"])
        if name == "edit_file":
            return tools.edit_file(
                working_dir, arguments["path"], arguments["old_str"], arguments["new_str"]
            )
        if name == "bash":
            result = tools.run_bash(working_dir, arguments["command"])
            return json.dumps(result)
        if name == "web_fetch":
            return tools.web_fetch(arguments["url"])
        if name == "web_search":
            return json.dumps(
                tools.web_search(arguments["query"], tavily_api_key=cfg.tavily_api_key)
            )
        return f"error: unknown tool '{name}'"
    except tools.PathJailViolation as exc:
        return f"error: {exc}"
    except Exception as exc:  # noqa: BLE001 - tool errors must reach the model, not crash the worker
        return f"error: {exc}"


def _make_chat_fn(cfg, model: str):
    """Bind one configured model into the single-argument callable run_job wants."""

    def chat_fn(messages: list[dict]) -> dict:
        return ollama_client.chat(cfg.ollama_host, model, messages, TOOL_SCHEMAS)

    return chat_fn


def main(job_id: str) -> None:
    cfg = config.load()
    conn = db.connect(cfg.db_path)
    pid = os.getpid()
    db.set_pid(conn, job_id, pid)

    try:
        job = db.get_job(conn, job_id)
        if job is None:
            return

        if not cfg.tiers:
            db.update_status(
                conn, job_id, "error",
                result_summary=(
                    "no model tiers configured - create "
                    f"{config.default_config_path()}"
                ),
            )
            return

        cfg.jobs_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.jobs_log_dir / f"{job_id}.log"
        db.update_status(conn, job_id, "running", transcript_path=str(log_path))

        base_tier, *escalation = cfg.tiers
        if not ollama_client.model_is_pulled(cfg.ollama_host, base_tier.model):
            ollama_client.pull_model(cfg.ollama_host, base_tier.model)

        run_job(
            conn,
            job_id,
            job["task"],
            job["working_dir"],
            log_path,
            chat_fn=_make_chat_fn(cfg, base_tier.model),
            max_iterations=cfg.max_iterations,
            max_wall_clock_seconds=cfg.max_wall_clock_seconds,
            watchdog_max_retries=cfg.watchdog_max_retries,
            execute_tool_fn=lambda name, arguments: execute_tool_call(
                job["working_dir"], name, arguments, cfg
            ),
            escalation_tiers=[
                (tier.threshold_tokens, tier.name, _make_chat_fn(cfg, tier.model))
                for tier in escalation
            ],
        )
    except ollama_client.OllamaError as exc:
        db.update_status(conn, job_id, "error", result_summary=str(exc))
    except Exception:
        db.update_status(conn, job_id, "error", result_summary=traceback.format_exc())
    finally:
        next_job_id = db.claim_next_queued_job(conn, pid)
        conn.close()
        if next_job_id:
            spawn_worker(next_job_id)


def spawn_worker(job_id: str) -> int:
    cfg = config.load()
    cfg.jobs_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.jobs_log_dir / f"{job_id}.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            # -m, not a file path: executing the file directly would put
            # handyman/ on sys.path instead of the repo root, breaking the
            # package-absolute imports at the top of this module.
            [sys.executable, "-m", "handyman.worker", job_id],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **procutil.detached_kwargs(),
        )
    return process.pid


if __name__ == "__main__":
    main(sys.argv[1])
