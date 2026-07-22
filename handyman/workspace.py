"""Keeping a delegation target in a state where results can be trusted.

Every guard here exists because a real run produced a misleading result.
In each case a tooling problem was first read as the model getting
something wrong, which is the expensive kind of mistake: it sends you
rewriting instructions when the instructions were fine.

They live in the package, rather than in whatever script someone writes
around it, so nobody has to rediscover them.
"""

import ast
import json
import os
import pathlib
import shutil
import subprocess
import time

WORKER_MARKER = "handyman.worker"


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------

def _processes() -> list[dict]:
    """Every running process with its parent and command line.

    Windows-only for now; on other platforms `ps` output would be parsed
    instead. Returns an empty list rather than raising, because process
    inspection is a diagnostic aid and must never break a caller.
    """
    if os.name != "nt":
        try:
            out = subprocess.run(["ps", "-eo", "pid,ppid,command"],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return []
        rows = []
        for line in out.splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) == 3:
                rows.append({"ProcessId": int(parts[0]), "ParentProcessId": int(parts[1]),
                             "CommandLine": parts[2]})
        return rows
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Select-Object "
             "ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out or "[]")
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def processes_touching(needle: str) -> list[dict]:
    needle = needle.lower()
    return [p for p in _processes()
            if p.get("CommandLine") and needle in p["CommandLine"].lower()]


def kill_processes(needle: str) -> tuple[list[int], list[int]]:
    """Kill everything whose command line mentions `needle`; verify after.

    Pattern-matching a shell script's name does not reach the Python
    children it spawned. Two builds once ran concurrently against one rate
    limit because a kill silently did nothing, so this reports what
    survived instead of assuming success.
    """
    targets = [p["ProcessId"] for p in processes_touching(needle)]
    for pid in targets:
        try:
            if os.name == "nt":
                subprocess.run(["powershell", "-NoProfile", "-Command",
                                f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                               capture_output=True, timeout=15)
            else:
                os.kill(pid, 9)
        except Exception:
            pass
    time.sleep(1)
    survivors = [p["ProcessId"] for p in processes_touching(needle)]
    return targets, survivors


def orphan_workers() -> list[dict]:
    """Workers whose parent is gone.

    Workers are detached so they survive a controller crash, which also
    means stopping the controller never stops them. Individually harmless;
    collectively they hold files open, which is what makes a later
    workspace reset fail while appearing to succeed.
    """
    procs = _processes()
    alive = {p.get("ProcessId") for p in procs}
    return [p for p in procs
            if p.get("CommandLine") and WORKER_MARKER in p["CommandLine"]
            and p.get("ParentProcessId") not in alive]


# --------------------------------------------------------------------------
# Workspace lifecycle
# --------------------------------------------------------------------------

def clear_bytecode(root) -> int:
    """Remove __pycache__ trees.

    A regenerated module leaves stale bytecode behind and the test runner
    will happily import it. That once looked like three genuine test
    failures in a module that was already correct.
    """
    removed = 0
    for cache in pathlib.Path(root).rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    return removed


def reset(target, tests_from=None, deps=("pytest",)) -> None:
    """Return a workspace to a known-empty state, and prove it worked.

    Deleting a directory can report success while leaving files behind if
    something still holds them open, so this kills holders first, retries,
    and raises rather than letting a later run inherit stale modules. It
    also rebuilds the virtual environment: a reset that removed one left
    the test command unable to run at all, and the model was blamed.
    """
    target = pathlib.Path(target).resolve()
    kill_processes(str(target))

    for attempt in range(5):
        shutil.rmtree(target, ignore_errors=True)
        if not target.exists():
            break
        time.sleep(2 ** attempt)
    if target.exists():
        raise RuntimeError(
            f"{target} could not be removed - something still holds it open. "
            "`handyman ps --orphans` will show detached workers."
        )

    (target / "tests").mkdir(parents=True)
    if tests_from:
        for f in pathlib.Path(tests_from).glob("*.py"):
            shutil.copy(f, target / "tests" / f.name)

    if deps:
        subprocess.run(["uv", "venv", "--quiet"], cwd=str(target),
                       check=True, capture_output=True)
        env = dict(os.environ, VIRTUAL_ENV=str(target / ".venv"))
        subprocess.run(["uv", "pip", "install", "--quiet", *deps],
                       cwd=str(target), env=env, check=True, capture_output=True)


def check_syntax(root) -> list[str]:
    """Files that were written but do not parse.

    Both models and humans produce these: a model writing its own
    deliberation mid-expression, a person mangling an escape through a
    shell. Either way the suite fails to collect and the cause is not
    visible in the test output.
    """
    broken = []
    for f in pathlib.Path(root).rglob("*.py"):
        if ".venv" in f.parts or "__pycache__" in f.parts:
            continue
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            broken.append(f"{f}: {exc}")
    return broken


def lint_brief(text: str) -> list[str]:
    """Problems in a task description, before it costs a run to discover.

    Deliberately small. A richer check that counted files against phrases
    like "three files" was tried and removed: briefs mention files in too
    many roles, and every rule that fixed one false positive created
    another.
    """
    problems = []
    if "\\n" in text or '\\"' in text:
        problems.append(
            "contains literal backslash escapes - shell quoting has probably "
            "mangled this, and the model will write the backslashes verbatim"
        )
    lowered = text.lower()
    if not any(phrase in lowered for phrase in
               ("stop", "you are finished", "then finish")):
        problems.append(
            "no explicit finish line - the single largest measured effect on "
            "whether a job ends promptly; add 'then stop'"
        )
    if len(text.strip()) < 20:
        problems.append("too short to describe a task")
    return problems
