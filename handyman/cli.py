"""Command line for delegating work and for keeping the loop honest.

Beyond the three job verbs, this carries the surrounding machinery that
otherwise gets rebuilt by hand around every use: running a task to
completion and reporting what it changed, running a queue of them,
checking a workspace before trusting a result, and finding the detached
workers that make a later reset fail. Those were all learned the hard
way, and they belong here rather than in whatever script wraps this.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

from handyman import config, server, workspace

POLL_SECONDS = 5
TERMINAL = ("done", "error", "incomplete", "timeout", "canceled")


def _emit(result: dict) -> int:
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


def _snapshot(root: pathlib.Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and not {".git", "__pycache__", ".venv"} & set(p.parts):
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime)
    return out


def cmd_run(args) -> int:
    """Delegate one task, wait for it, and report enough to accept or reject.

    Written because polling, diffing and testing by hand around every
    delegation is most of the cost of using this at all.
    """
    task = pathlib.Path(args.task).read_text(encoding="utf-8") \
        if pathlib.Path(args.task).exists() else args.task
    workdir = pathlib.Path(args.working_dir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    for problem in workspace.lint_brief(task):
        print(f"  brief warning: {problem}")

    before = _snapshot(workdir)
    started = time.monotonic()

    result = server.gemma_delegate(task, str(workdir),
                                   allow_hosted=getattr(args, "allow_hosted", False))
    job_id = result.get("job_id")
    if not job_id:
        return _emit(result)
    print(f"job {job_id[:8]} ...", flush=True)

    status, last, unreadable = "running", None, 0
    while time.monotonic() - started < args.timeout:
        time.sleep(POLL_SECONDS)
        state = server.gemma_check(job_id)
        # A status that cannot be read is not the same as a job still
        # running. Conflating them once meant polling for 57 minutes after
        # the job had already failed.
        if "error" in state and "status" not in state:
            unreadable += 1
            if unreadable >= 3:
                print(f"  status unreadable: {state['error'][:160]}")
                break
            continue
        unreadable = 0
        status = state.get("status", "unknown")
        marker = (state.get("iteration"), state.get("last_action"))
        if marker != last:
            print(f"  [{time.monotonic() - started:5.0f}s] "
                  f"iter={marker[0]} {marker[1]}", flush=True)
            last = marker
        if status in TERMINAL:
            break

    elapsed = time.monotonic() - started
    state = server.gemma_check(job_id)
    print(f"\nstatus  : {status}")
    print(f"elapsed : {elapsed:.0f}s")
    if state.get("result_summary"):
        print(f"summary : {state['result_summary'][:300]}")

    after = _snapshot(workdir)
    created = sorted(set(after) - set(before))
    changed = sorted(k for k in set(after) & set(before) if after[k] != before[k])
    print(f"created : {created or '-'}")
    print(f"modified: {changed or '-'}")

    workspace.clear_bytecode(workdir)
    for broken in workspace.check_syntax(workdir):
        print(f"  does not parse: {broken}")

    ok = status == "done"
    if args.test:
        print(f"\n$ {args.test}")
        proc = subprocess.run(args.test, shell=True, cwd=str(workdir),
                              capture_output=True, text=True)
        print("\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:]))
        ok = ok and proc.returncode == 0
        print(f"tests   : {'PASS' if proc.returncode == 0 else 'FAIL'}")
    return 0 if ok else 1


def cmd_batch(args) -> int:
    """Run a queue of tasks in order, logging each outcome.

    Strictly sequential: a local model serves one request at a time, and
    overlapping runs only thrash. A failure does not stop the queue -
    later tasks are usually independent, and a full pass says more about
    consistency than halting at the first problem does.
    """
    entries = []
    for line in pathlib.Path(args.queue).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            task, _, test = line.partition("|")
            entries.append((task.strip(), test.strip()))

    results = []
    for task, test in entries:
        ns = argparse.Namespace(task=task, working_dir=args.working_dir,
                                test=(test if test and test != "-" else None),
                                timeout=args.timeout,
                                allow_hosted=getattr(args, "allow_hosted", False))
        started = time.monotonic()
        code = cmd_run(ns)
        results.append((task, code == 0, time.monotonic() - started))
        print("-" * 60, flush=True)

    print("\n=== summary ===")
    for task, ok, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {secs:6.0f}s  {task}")
    return 0 if all(ok for _, ok, _ in results) else 1


def cmd_doctor(args) -> int:
    """Check that a run could produce a trustworthy result at all."""
    problems = []
    cfg = None
    try:
        cfg = config.load()
        print(f"config   : {config.default_config_path()}")
        if not cfg.tiers:
            problems.append("no model tiers configured - run `handyman setup`")
        else:
            t = cfg.tiers[0]
            print(f"model    : {t.model} ({len(cfg.tiers)} tier(s))")
    except Exception as exc:
        problems.append(f"config will not load: {exc}")

    if cfg:
        print(f"data dir : {cfg.db_path.parent}")
        hosted = bool(config.api_key_for(cfg))
        print(f"provider : {'hosted (api key set)' if hosted else 'local'}")
        if not hosted:
            try:
                import requests
                requests.get(f"{cfg.ollama_host}/api/tags", timeout=5).raise_for_status()
                print(f"server   : reachable at {cfg.ollama_host}")
            except Exception:
                problems.append(f"model server unreachable at {cfg.ollama_host}")
        elif not cfg.api_key_env or not config.api_key_for(cfg):
            problems.append(f"api_key_env is {cfg.api_key_env!r} but that variable is empty")

    if args.workspace:
        ws = pathlib.Path(args.workspace)
        if not (ws / ".venv").exists():
            problems.append(f"{ws} has no virtual environment - its test command would fail")
        if not list((ws / "tests").glob("test_*.py")) if (ws / "tests").exists() else True:
            problems.append(f"{ws} has no tests - nothing would verify a result")
        for broken in workspace.check_syntax(ws):
            problems.append(f"does not parse: {broken}")

    orphans = workspace.orphan_workers()
    if orphans:
        problems.append(f"{len(orphans)} orphaned worker(s) - they hold files open; "
                        "`handyman ps --kill-orphans`")

    print()
    for p in problems:
        print(f"  PROBLEM: {p}")
    print("all clear" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def cmd_ps(args) -> int:
    orphans = workspace.orphan_workers()
    for o in orphans:
        print(f"  {o['ProcessId']}  {o.get('CommandLine', '')[:100]}")
    print(f"{len(orphans)} orphaned worker(s)")
    if args.kill_orphans and orphans:
        killed, survivors = workspace.kill_processes(workspace.WORKER_MARKER)
        print(f"killed {len(killed)}; still alive: {survivors or 'none'}")
        return 1 if survivors else 0
    return 0


def cmd_reset(args) -> int:
    workspace.reset(args.directory, tests_from=args.tests_from,
                    deps=tuple(args.deps.split(",")) if args.deps else ())
    print(f"reset: {args.directory}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="handyman", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("delegate", help="submit a task and return immediately")
    d.add_argument("task"); d.add_argument("working_dir")
    d.add_argument("--allow-hosted", action="store_true",
                   help="permit a hosted model when local is busy or down; "
                        "this sends the task off the machine")

    c = sub.add_parser("check", help="status of a job"); c.add_argument("job_id")
    x = sub.add_parser("cancel", help="ask a job to stop"); x.add_argument("job_id")

    r = sub.add_parser("run", help="submit a task, wait, and report what changed")
    r.add_argument("task", help="task text, or a path to a file containing it")
    r.add_argument("working_dir")
    r.add_argument("--test", help="command to run afterwards, in the working dir")
    r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--allow-hosted", action="store_true")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="run a queue of tasks in order")
    b.add_argument("queue", help="one 'task-file|test-command' per line")
    b.add_argument("working_dir")
    b.add_argument("--timeout", type=int, default=3600)
    b.add_argument("--allow-hosted", action="store_true")
    b.set_defaults(func=cmd_batch)

    doc = sub.add_parser("doctor", help="check that results could be trusted")
    doc.add_argument("--workspace")
    doc.set_defaults(func=cmd_doctor)

    ps = sub.add_parser("ps", help="find detached workers with no live parent")
    ps.add_argument("--kill-orphans", action="store_true")
    ps.set_defaults(func=cmd_ps)

    rs = sub.add_parser("reset", help="return a workspace to a verified-clean state")
    rs.add_argument("directory")
    rs.add_argument("--tests-from")
    rs.add_argument("--deps", default="pytest")
    rs.set_defaults(func=cmd_reset)
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    if args.cmd == "delegate":
        return _emit(server.gemma_delegate(args.task, args.working_dir,
                                           allow_hosted=args.allow_hosted))
    if args.cmd == "check":
        return _emit(server.gemma_check(args.job_id))
    if args.cmd == "cancel":
        return _emit(server.gemma_cancel(args.job_id))
    return 1


def entrypoint() -> None:
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    entrypoint()
