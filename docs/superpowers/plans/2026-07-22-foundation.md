# OSS Generalization — Plan A: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the working single-machine tool into an installable, config-file-driven Python package that is verified to run on Windows, macOS, and Linux — with no change to what the tool does for the user.

**Architecture:** Four sequential tasks. Task 1 moves the flat modules into a `handyman/` package with a `pyproject.toml`. Task 2 extracts every OS-specific branch into `handyman/procutil.py` and adds the missing POSIX process-tree kill. Task 3 replaces the env-var constants in `config.py` with a loaded YAML config file supporting 1-3 tiers. Task 4 adds the GitHub Actions matrix that proves the cross-platform claim. `worker.run_job` is not modified by any task.

**Tech Stack:** Python 3.11+, pytest, PyYAML, sqlite3 (stdlib), GitHub Actions.

**Source spec:** `docs/superpowers/specs/2026-07-22-oss-generalization-design.md`

**Execution context (changed 2026-07-22):** this is a **standalone repo**,
not a branch of gemma-agent. It was seeded with gemma-agent's proven
runtime (7 modules, 127 tests) and is otherwise inert — nothing here is
running, no MCP server points at it. That matters: the earlier in-place
attempt kept breaking the tool being used to do the work. Here the code
under construction and the agent doing the construction are fully
separate, so restructuring is safe at any point.

**Delegation model:** implementation is delegated to the local model
(gemma-12b via gemma-agent, driven through a frozen foreground runner);
the controller reviews and audits every step. Two measured constraints
shape how tasks are written:

- The model copies whole files with perfect fidelity (164 lines across
  two files, zero drift) but **spiralled and failed on a surgical edit**
  requiring an exact 32-line `old_str` from a 237-line file (1183s, no
  work done). **Prefer whole-file writes over in-place edits.**
- Throughput is ~5.5s fixed overhead + ~22 tok/s, so a ~900-line file is
  minutes, not seconds. Steps are sized accordingly.

## Global Constraints

- **`handyman` is a working placeholder, NOT a confirmed name.** Use these exact values: PyPI distribution `handyman-agent`; import package and CLI command both `handyman`; env prefix `HANDYMAN_`; config dir `~/.config/handyman/`. The bare name `handyman` is taken on PyPI by an unrelated project, which is why the distribution name is hyphenated while the import name is not — do not "fix" that mismatch. Because the name may still change, keep it out of log strings, docstrings, and error messages, so a rename stays mechanical.
- **Python floor: 3.11.** `pyproject.toml` sets `requires-python = ">=3.11"`.
- **Env var prefix changes `GEMMA_` → `HANDYMAN_`.** No backwards compatibility with the old prefix — this is a pre-release personal tool with one user.
- **`worker.run_job` must not be modified.** Its `escalation_tiers` parameter is already the correct seam. Any task that appears to need a change inside `run_job` is a signal the task is wrong.
- **`tests/test_db.py`'s 30 tests must stay green with their logic unedited.** Import lines may change (a package restructure forces that); assertions, fixtures, and test bodies may not. They cover concurrency and stranded-job invariants hardened over 7 review rounds. If a change requires touching an assertion, the invariant moved — stop and flag it.
- **The first configured tier's `name` must equal `db.BASE_TIER` (`"small"`).** `run_job` initializes `current_tier = db.BASE_TIER`, and `db.try_claim_with_cap` blocks claims when running jobs disagree on tier. A first tier named anything else silently breaks cross-job tier blocking.
- **Delete `MAX_TOTAL_TOKENS`.** It has been dead config since the token cap was dropped from `run_job`'s signature; do not carry it into the new schema.
- **Full suite must pass at every commit.** Baseline is 127 tests.

---

### Task 1: Package restructure and pyproject

Moves seven flat modules into a `handyman/` package and makes the project installable. Purely structural — no behavior changes. Paths (`jobs.db`, `jobs/`) must resolve exactly where they do today; Task 3 moves them.

**Files:**
- Create: `pyproject.toml`
- Create: `handyman/__init__.py`
- Create: `README.md` (required — `pyproject.toml` declares it, so the install fails without it)
- Create: `LICENSE`
- Move: `cli.py`, `config.py`, `db.py`, `ollama_client.py`, `server.py`, `tools.py`, `worker.py` → `handyman/`
- Modify: `tests/conftest.py`
- Modify: all files in `tests/` (import statements)
- Delete: `requirements.txt` (superseded by `pyproject.toml`)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: the `handyman` package. Every later task and both later plans import as `from handyman import config, db, procutil`. Console entry point `handyman = "handyman.cli:entrypoint"`. Worker subprocesses are spawned as `python -m handyman.worker <job_id>`.

- [ ] **Step 1: Move the modules**

```bash
mkdir handyman
git mv cli.py config.py db.py ollama_client.py server.py tools.py worker.py handyman/
```

- [ ] **Step 2: Create the package init**

Create `handyman/__init__.py`:

```python
"""Delegate grunt work to a local LLM over MCP."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "handyman-agent"
version = "0.1.0"
description = "Delegate grunt work to a local LLM over MCP"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "mcp>=1.2.0",
    "requests>=2.31.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[project.scripts]
handyman = "handyman.cli:entrypoint"

[tool.setuptools]
packages = ["handyman"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3b: Create README.md and LICENSE**

`pyproject.toml` declares `readme = "README.md"`; without the file, `pip install -e .` fails with a build error. The README also carries the path-jail disclosure the spec requires (the "docs-only safety gate" decision).

Create `README.md`:

```markdown
# handyman

> **Note:** `handyman` is a working placeholder and may change before release.

Delegate grunt work to a local LLM, so your expensive agent doesn't spend
tokens on it. Runs as an MCP server: your agent calls `gemma_delegate`,
a local model does the work in a background process, and your agent
collects the result later.

## Status

Pre-release. Works on one machine; being generalized for wider use.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with a tool-calling-capable model

## Install

```bash
pip install -e .
```

## Safety: what the path jail does and does not cover

Delegated jobs run with real filesystem and shell access. Two things
constrain them, and it is important to understand the limits of both:

**What is enforced.** File tools (`read_file`, `write_file`, `edit_file`)
resolve every path against the job's `working_dir` and refuse paths that
escape it, including via symlinks and Windows directory junctions.

**What is NOT enforced.** `run_bash` executes arbitrary shell commands.
It runs *with `working_dir` as its current directory*, but nothing stops
a command from using an absolute path, deleting files, making network
requests, or otherwise touching anything your user account can touch.
The path jail constrains the file tools, not the shell.

Treat a delegated job as running with your full user privileges, because
it does. Only delegate to a working directory you would be comfortable
handing to an unattended script, and read the task you are delegating.

## License

MIT — see `LICENSE`.
```

Create `LICENSE` with the standard MIT text, copyright `2026 Sachin Gulati`.

- [ ] **Step 4: Fix intra-package imports**

In each of `handyman/cli.py`, `handyman/server.py`, `handyman/worker.py`, `handyman/tools.py`, replace bare module imports with package-absolute ones. The exact edits:

`handyman/cli.py` line 4: `import server` → `from handyman import server`

`handyman/server.py`: `import config` → `from handyman import config`; `import db` → `from handyman import db`; `import worker` → `from handyman import worker` (only if present — check with grep below).

`handyman/worker.py`: `import config` → `from handyman import config`; `import db` → `from handyman import db`; `import ollama_client` → `from handyman import ollama_client`; `import tools` → `from handyman import tools`.

Find every remaining bare import:

```bash
grep -rn "^import \(config\|db\|server\|worker\|tools\|ollama_client\)\|^from \(config\|db\|server\|worker\|tools\|ollama_client\) import" handyman/
```

Expected after the edits: no output.

- [ ] **Step 5: Keep paths pointing at the repo root**

`handyman/config.py` currently has `PROJECT_ROOT = Path(__file__).resolve().parent`, which now resolves to `handyman/` instead of the repo root. Restore the old location so `jobs.db` and `jobs/` do not move in this task.

In `handyman/config.py`, replace:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
```

with:

```python
# .parent.parent because this module now lives inside the handyman/
# package: parent is handyman/, parent.parent is the repo root, which is
# where jobs.db and jobs/ already live. Task 3 replaces this entirely
# with a platform-appropriate user data directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

- [ ] **Step 6: Fix the worker spawn to use module execution**

`python handyman/worker.py <id>` cannot resolve `from handyman import config`, because running a file directly puts `handyman/` on `sys.path` rather than the repo root. Switch to `-m`.

In `handyman/worker.py`, in `spawn_worker`, replace:

```python
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), job_id],
```

with:

```python
        subprocess.Popen(
            # -m, not a file path: executing the file directly would put
            # handyman/ on sys.path instead of the repo root, breaking the
            # package-absolute imports at the top of this module.
            [sys.executable, "-m", "handyman.worker", job_id],
```

- [ ] **Step 7: Add the CLI entry point function**

`pyproject.toml` declares `handyman.cli:entrypoint`, which does not exist yet. Append to `handyman/cli.py`, replacing the existing `if __name__ == "__main__":` block:

```python
def entrypoint() -> None:
    """Console-script entry point declared in pyproject.toml."""
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    entrypoint()
```

- [ ] **Step 8: Update test imports**

`tests/conftest.py` already inserts the repo root on `sys.path`, so `import handyman.x` resolves without installation. Keep it, but document why:

Replace the whole of `tests/conftest.py` with:

```python
import sys
from pathlib import Path

# Put the repo root on sys.path so `from handyman import ...` resolves when
# the suite is run without `pip install -e .` (the common local case).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Then update every test module's imports. Find them:

```bash
grep -rn "import \(config\|db\|server\|worker\|tools\|ollama_client\|cli\)" tests/
grep -rn "[\"']\(config\|db\|server\|worker\|tools\|ollama_client\|cli\)\." tests/
```

Both greps matter. Imports are **not** all at module level — several tests
import inside the function body (indented), so an `^`-anchored pattern
misses them. The second grep catches string-based patching such as
`monkeypatch.setattr("server._spawn_worker", ...)`, where the module path
lives in a string literal and no import statement appears at all.

For each hit, rewrite `import X` as `from handyman import X`. The module is still referenced as `X.` throughout each test body, so no other edits are needed.

- [ ] **Step 9: Update the spawn_worker test**

The `spawn_worker` stdout/stderr test lives in `tests/test_worker_dispatch.py`, not `test_worker.py`. Check whether it asserts on the `Popen` argv (which changed in Step 6) — it may only assert on the file handles, in which case nothing needs updating. Find it:

```bash
grep -rn "worker.py\|sys.executable\|argv" tests/ | head -20
```

Update the expected argv to `[sys.executable, "-m", "handyman.worker", job_id]`. Leave the stdout/stderr assertions untouched — that fix (`9cb94d2`) is load-bearing.

- [ ] **Step 10: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: `127 passed`

If any test fails with `ModuleNotFoundError`, a bare import was missed — re-run the greps from Steps 4 and 8.

- [ ] **Step 11: Verify the package actually installs and the entry point works**

```bash
VIRTUAL_ENV=.venv uv pip install -e .
.venv/Scripts/python.exe -c "import handyman; print(handyman.__version__)"
.venv/Scripts/handyman.exe check bogus
```

Note: this project's `.venv` is uv-managed and has **no `pip`** — `python -m pip` fails with "No module named pip". Use `uv pip install`. (Task 4's CI runners do have pip, so the workflow file is unaffected.)

Expected: `0.1.0`

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "Restructure into an installable handyman package

Moves the seven flat modules under handyman/ and adds pyproject.toml with
a console entry point. Worker subprocesses now spawn via -m so
package-absolute imports resolve. Paths still resolve to the repo root;
Task 3 moves them to a user data directory.

handyman is a working placeholder, not a confirmed name."
```

---

### Task 2: Cross-platform process utilities

Extracts every OS-specific branch into one module and closes the real gap: `run_bash`'s timeout kill is Windows-only today, so on macOS and Linux a timed-out command's grandchild survives and `communicate()` blocks until it exits on its own.

**Files:**
- Create: `handyman/procutil.py`
- Create: `tests/test_procutil.py`
- Modify: `handyman/db.py` (remove `is_pid_alive` body, re-export)
- Modify: `handyman/tools.py:53-83` (`run_bash`)
- Modify: `handyman/worker.py` (`spawn_worker` creationflags)

**Interfaces:**
- Consumes: the `handyman` package from Task 1.
- Produces: `procutil.is_pid_alive(pid: int) -> bool`, `procutil.kill_process_tree(pid: int) -> None`, `procutil.process_group_kwargs() -> dict`, `procutil.detached_kwargs() -> dict`. `db.is_pid_alive` remains importable as a re-export so `tests/test_db.py` stays untouched.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_procutil.py`:

```python
import os
import subprocess
import sys
import time

import pytest

from handyman import procutil


def test_is_pid_alive_true_for_current_process():
    assert procutil.is_pid_alive(os.getpid()) is True


def test_is_pid_alive_false_for_zero():
    assert procutil.is_pid_alive(0) is False


def test_process_group_kwargs_posix_starts_new_session(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert procutil.process_group_kwargs() == {"start_new_session": True}


def test_process_group_kwargs_windows_is_empty(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert procutil.process_group_kwargs() == {}


def test_detached_kwargs_windows_uses_create_no_window(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    kwargs = procutil.detached_kwargs()
    assert "creationflags" in kwargs


def test_detached_kwargs_posix_starts_new_session(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert procutil.detached_kwargs() == {"start_new_session": True}


def test_kill_process_tree_kills_a_real_grandchild():
    """The child is a shell; the sleeping python is its grandchild.

    Killing only the immediate child leaves the grandchild running, which
    is exactly the bug this function exists to prevent.
    """
    command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **procutil.process_group_kwargs(),
    )
    try:
        procutil.kill_process_tree(process.pid)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.1)
        assert process.poll() is not None, "process tree survived the kill"
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def test_kill_process_tree_is_quiet_for_a_dead_pid():
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"], **procutil.process_group_kwargs()
    )
    process.wait()
    procutil.kill_process_tree(process.pid)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_procutil.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'handyman.procutil'`

- [ ] **Step 3: Write procutil.py**

Create `handyman/procutil.py`:

```python
"""Cross-platform process helpers.

Every OS-specific branch in the project lives here, so the rest of the
code stays platform-agnostic and the branches can be exercised in tests
by monkeypatching os.name in exactly one place.
"""

import os
import signal
import subprocess


def is_pid_alive(pid: int) -> bool:
    """Best-effort check for whether a process id is still running."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            # OpenProcess can succeed for a process that has already
            # exited but whose object hasn't been torn down yet (e.g. a
            # subprocess.Popen still holding its own handle) - the exit
            # code is the only reliable liveness signal.
            exit_code = ctypes.c_ulong(0)
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_group_kwargs() -> dict:
    """Popen kwargs that make the child lead its own process group.

    Required on POSIX for kill_process_tree to have a group to signal.
    Windows needs nothing here: taskkill /T walks the OS's own
    parent-child bookkeeping instead.
    """
    if os.name == "nt":
        return {}
    return {"start_new_session": True}


def detached_kwargs() -> dict:
    """Popen kwargs for a long-lived background worker.

    CREATE_NO_WINDOW suppresses the console window python.exe (a
    console-subsystem binary) would otherwise pop up. Windows documents
    DETACHED_PROCESS as mutually exclusive with CREATE_NO_WINDOW -
    combining them produced a brief window flash instead of reliable
    suppression, so only CREATE_NO_WINDOW is used.
    """
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def kill_process_tree(pid: int) -> None:
    """Kill a process and everything it spawned.

    process.kill() alone only kills the immediate child. With shell=True
    that child is "cmd.exe /c <command>" (Windows) or "/bin/sh -c
    <command>" (POSIX) - the actual command runs as a grandchild that
    survives, and a following communicate() would block until that orphan
    exits on its own.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            text=True,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_procutil.py -q`
Expected: `8 passed`

- [ ] **Step 5: Re-export is_pid_alive from db**

In `handyman/db.py`, delete the entire `is_pid_alive` function body (lines 11-41) and replace it with a re-export directly below the existing imports:

```python
from handyman.procutil import is_pid_alive  # re-exported: db.is_pid_alive is a documented call site
```

Remove the now-unused `import os` from `handyman/db.py` only if nothing else in the file uses it:

```bash
grep -n "os\." handyman/db.py
```

If that returns no output, delete the `import os` line.

- [ ] **Step 6: Verify db tests still pass unmodified**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db.py -q`
Expected: `30 passed` — and `git diff --stat tests/test_db.py` must show no changes.

- [ ] **Step 7: Rewrite run_bash to use procutil**

In `handyman/tools.py`, add `from handyman import procutil` to the imports, then replace `run_bash` (lines 53-83) entirely:

```python
def run_bash(working_dir: str, command: str, timeout: int = 60) -> dict:
    root = Path(working_dir).resolve()
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **procutil.process_group_kwargs(),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return {"stdout": stdout, "stderr": stderr, "return_code": process.returncode}
    except subprocess.TimeoutExpired:
        procutil.kill_process_tree(process.pid)
        stdout, stderr = process.communicate()
        stderr = (stderr or "") + f"\n[timed out after {timeout}s]"
        return {"stdout": stdout or "", "stderr": stderr, "return_code": -1}
```

- [ ] **Step 8: Add a portable timeout-enforcement test**

Append to `tests/test_tools_bash.py`:

```python
def test_run_bash_timeout_kills_the_grandchild_promptly(tmp_path):
    """A timed-out command must not leave communicate() blocking on an orphan.

    The 30s sleep is far longer than the 1s timeout, so if the grandchild
    survived, this test would take ~30s instead of ~1s.
    """
    command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    started = time.monotonic()
    result = tools.run_bash(str(tmp_path), command, timeout=1)
    elapsed = time.monotonic() - started

    assert result["return_code"] == -1
    assert "[timed out after 1s]" in result["stderr"]
    assert elapsed < 15, f"timeout kill did not release communicate() (took {elapsed:.1f}s)"
```

Add `import sys` and `import time` to that file's imports if not already present.

- [ ] **Step 9: Run the bash tool tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_bash.py -q`
Expected: `6 passed`

- [ ] **Step 10: Use detached_kwargs in spawn_worker**

In `handyman/worker.py`, add `from handyman import procutil` to the imports and replace the `creationflags` lines in `spawn_worker`:

```python
def spawn_worker(job_id: str) -> None:
    log_path = config.JOBS_LOG_DIR / f"{job_id}.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            # -m, not a file path: executing the file directly would put
            # handyman/ on sys.path instead of the repo root, breaking the
            # package-absolute imports at the top of this module.
            [sys.executable, "-m", "handyman.worker", job_id],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **procutil.detached_kwargs(),
        )
```

- [ ] **Step 11: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: `136 passed` (127 baseline + 8 procutil + 1 timeout test)

If `test_spawn_worker_redirects_stdout_and_stderr_to_job_log` fails on the `creationflags` kwarg, update its assertion to accept the kwargs dict from `procutil.detached_kwargs()` rather than a literal `creationflags` value.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "Extract cross-platform process helpers, fix POSIX timeout kill

run_bash's timeout kill was Windows-only: on macOS and Linux a timed-out
command's grandchild survived and communicate() blocked until it exited
on its own. POSIX now gets start_new_session plus killpg/SIGKILL.

is_pid_alive moves to procutil and is re-exported from db so
tests/test_db.py stays untouched."
```

---

### Task 3: Config file with 1-3 configurable tiers

Replaces the three hardcoded model constants with a loaded YAML file, so tier count becomes data rather than code. This is what the setup wizard (Plan B) writes and what the hosted provider (Plan C) extends.

**Files:**
- Rewrite: `handyman/config.py`
- Create: `tests/test_config_file.py`
- Rewrite: `tests/test_config.py`
- Modify: `handyman/worker.py` (`main`, and delete `_default_chat_fn`/`_mid_chat_fn`/`_big_chat_fn`)

**Interfaces:**
- Consumes: `handyman` package (Task 1).
- Produces:
  - `config.Tier` — frozen dataclass with `name: str`, `model: str`, `threshold_tokens: int`.
  - `config.Config` — frozen dataclass with `tiers: list[Tier]`, `ollama_host: str`, `max_concurrent_jobs: int`, `max_iterations: int`, `max_wall_clock_seconds: int`, `watchdog_max_retries: int`, `tavily_api_key: str | None`, `db_path: Path`, `jobs_log_dir: Path`.
  - `config.load(path: Path | None = None) -> Config`
  - `config.default_config_path() -> Path`
  - `config.default_data_dir() -> Path`
  - `config.ConfigError` — raised on invalid config.
- Plan B's wizard writes a file matching this schema. Plan C adds a `hosted:` block to it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_file.py`:

```python
from pathlib import Path

import pytest

from handyman import config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_single_tier(tmp_path):
    path = _write(tmp_path, """
tiers:
  - name: small
    model: qwen3:8b
    threshold_tokens: 0
""")
    cfg = config.load(path)
    assert len(cfg.tiers) == 1
    assert cfg.tiers[0].name == "small"
    assert cfg.tiers[0].model == "qwen3:8b"
    assert cfg.tiers[0].threshold_tokens == 0


def test_loads_three_tiers_in_threshold_order(tmp_path):
    path = _write(tmp_path, """
tiers:
  - name: small
    model: m-small
    threshold_tokens: 0
  - name: mid
    model: m-mid
    threshold_tokens: 24000
  - name: big
    model: m-big
    threshold_tokens: 48000
""")
    cfg = config.load(path)
    assert [t.name for t in cfg.tiers] == ["small", "mid", "big"]
    assert [t.threshold_tokens for t in cfg.tiers] == [0, 24000, 48000]


def test_defaults_apply_when_file_omits_them(tmp_path):
    path = _write(tmp_path, """
tiers:
  - name: small
    model: m
    threshold_tokens: 0
""")
    cfg = config.load(path)
    assert cfg.ollama_host == "http://localhost:11434"
    assert cfg.max_concurrent_jobs == 3
    assert cfg.max_iterations == 40
    assert cfg.max_wall_clock_seconds == 1200
    assert cfg.watchdog_max_retries == 3
    assert cfg.tavily_api_key is None


def test_file_values_override_defaults(tmp_path):
    path = _write(tmp_path, """
ollama_host: http://elsewhere:1234
max_concurrent_jobs: 7
tiers:
  - name: small
    model: m
    threshold_tokens: 0
""")
    cfg = config.load(path)
    assert cfg.ollama_host == "http://elsewhere:1234"
    assert cfg.max_concurrent_jobs == 7


def test_env_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDYMAN_MAX_CONCURRENT_JOBS", "9")
    path = _write(tmp_path, """
max_concurrent_jobs: 7
tiers:
  - name: small
    model: m
    threshold_tokens: 0
""")
    cfg = config.load(path)
    assert cfg.max_concurrent_jobs == 9


def test_missing_file_yields_defaults_with_no_tiers(tmp_path):
    cfg = config.load(tmp_path / "does-not-exist.yaml")
    assert cfg.tiers == []
    assert cfg.max_concurrent_jobs == 3


def test_rejects_more_than_three_tiers(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, model: a, threshold_tokens: 0}
  - {name: b, model: b, threshold_tokens: 1}
  - {name: c, model: c, threshold_tokens: 2}
  - {name: d, model: d, threshold_tokens: 3}
""")
    with pytest.raises(config.ConfigError, match="1 to 3 tiers"):
        config.load(path)


def test_rejects_first_tier_not_named_small(tmp_path):
    """run_job initializes current_tier to db.BASE_TIER ('small') and
    try_claim_with_cap blocks claims when running jobs disagree on tier -
    so a differently-named first tier silently breaks tier blocking."""
    path = _write(tmp_path, """
tiers:
  - {name: tiny, model: a, threshold_tokens: 0}
""")
    with pytest.raises(config.ConfigError, match="must be named 'small'"):
        config.load(path)


def test_rejects_nonzero_first_threshold(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, model: a, threshold_tokens: 500}
""")
    with pytest.raises(config.ConfigError, match="threshold_tokens: 0"):
        config.load(path)


def test_rejects_non_ascending_thresholds(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, model: a, threshold_tokens: 0}
  - {name: mid, model: b, threshold_tokens: 40000}
  - {name: big, model: c, threshold_tokens: 20000}
""")
    with pytest.raises(config.ConfigError, match="ascending"):
        config.load(path)


def test_rejects_tier_missing_model(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, threshold_tokens: 0}
""")
    with pytest.raises(config.ConfigError, match="model"):
        config.load(path)


def test_tavily_key_prefers_handyman_specific_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "shared")
    monkeypatch.setenv("HANDYMAN_TAVILY_API_KEY", "specific")
    cfg = config.load(tmp_path / "none.yaml")
    assert cfg.tavily_api_key == "specific"


def test_tavily_key_falls_back_to_shared_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HANDYMAN_TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "shared")
    cfg = config.load(tmp_path / "none.yaml")
    assert cfg.tavily_api_key == "shared"


def test_default_config_path_is_platform_appropriate(monkeypatch):
    import os

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg")
    assert config.default_config_path() == Path("/tmp/xdg/handyman/config.yaml")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config_file.py -q`
Expected: FAIL — `AttributeError: module 'handyman.config' has no attribute 'ConfigError'`

- [ ] **Step 3: Rewrite config.py**

Replace the entire contents of `handyman/config.py`:

```python
"""Configuration: defaults, overlaid by a YAML file, overlaid by env vars.

The tier list is data rather than code so a machine can run 1, 2, or 3
model tiers without a code change. worker.main turns it into the
(threshold, name, chat_fn) triples run_job already consumes.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from handyman import db  # for BASE_TIER; db does not import config, so no cycle

APP_NAME = "handyman"  # config/data dir name; PyPI distribution is handyman-agent

DEFAULTS = {
    "ollama_host": "http://localhost:11434",
    "max_concurrent_jobs": 3,
    "max_iterations": 40,
    "max_wall_clock_seconds": 20 * 60,
    "watchdog_max_retries": 3,
}

# Env var name -> (config key, type). Env always wins over the file.
_ENV_OVERRIDES = {
    "HANDYMAN_OLLAMA_HOST": ("ollama_host", str),
    "HANDYMAN_MAX_CONCURRENT_JOBS": ("max_concurrent_jobs", int),
    "HANDYMAN_MAX_ITERATIONS": ("max_iterations", int),
    "HANDYMAN_MAX_WALL_CLOCK_SECONDS": ("max_wall_clock_seconds", int),
    "HANDYMAN_WATCHDOG_MAX_RETRIES": ("watchdog_max_retries", int),
}

# Deliberately not a second "small" literal: this value MUST equal
# db.BASE_TIER, and two constants that must agree will eventually drift.
BASE_TIER_NAME = db.BASE_TIER


class ConfigError(Exception):
    """Raised when a config file is present but unusable."""


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    threshold_tokens: int


@dataclass(frozen=True)
class Config:
    tiers: list[Tier]
    ollama_host: str
    max_concurrent_jobs: int
    max_iterations: int
    max_wall_clock_seconds: int
    watchdog_max_retries: int
    tavily_api_key: str | None
    db_path: Path
    jobs_log_dir: Path


def default_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_NAME / "config.yaml"


def default_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_NAME


def _parse_tiers(raw) -> list[Tier]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError("'tiers' must be a list")
    if not 1 <= len(raw) <= 3:
        raise ConfigError(f"config must define 1 to 3 tiers, got {len(raw)}")

    tiers = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"tier {index} must be a mapping")
        for key in ("name", "model"):
            if not entry.get(key):
                raise ConfigError(f"tier {index} is missing required key '{key}'")
        tiers.append(
            Tier(
                name=str(entry["name"]),
                model=str(entry["model"]),
                threshold_tokens=int(entry.get("threshold_tokens", 0)),
            )
        )

    # run_job initializes current_tier to db.BASE_TIER and
    # try_claim_with_cap refuses to admit a job whose tier differs from a
    # running one. If the first tier were named anything else, every job
    # would start on a tier no config entry matches and cross-job tier
    # blocking would silently stop working.
    if tiers[0].name != BASE_TIER_NAME:
        raise ConfigError(f"the first tier must be named '{BASE_TIER_NAME}', got '{tiers[0].name}'")
    if tiers[0].threshold_tokens != 0:
        raise ConfigError("the first tier must have threshold_tokens: 0")

    thresholds = [t.threshold_tokens for t in tiers]
    if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
        raise ConfigError(f"tier threshold_tokens must be strictly ascending, got {thresholds}")

    return tiers


def load(path: Path | None = None) -> Config:
    if path is None:
        path = Path(os.environ.get("HANDYMAN_CONFIG", default_config_path()))

    raw = {}
    if Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(f"{path} must contain a YAML mapping at the top level")
            raw = loaded

    values = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in raw:
            values[key] = raw[key]
    for env_var, (key, caster) in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            values[key] = caster(os.environ[env_var])

    data_dir = Path(os.environ.get("HANDYMAN_DATA_DIR", default_data_dir()))

    # A handyman-specific key wins so a user can point this tool at its own
    # key or quota; a plain TAVILY_API_KEY already in the environment is
    # accepted as a convenience so a second key isn't required.
    tavily = os.environ.get("HANDYMAN_TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY") or None

    return Config(
        tiers=_parse_tiers(raw.get("tiers")),
        tavily_api_key=tavily,
        db_path=Path(os.environ.get("HANDYMAN_DB_PATH", data_dir / "jobs.db")),
        jobs_log_dir=Path(os.environ.get("HANDYMAN_JOBS_LOG_DIR", data_dir / "jobs")),
        **values,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config_file.py -q`
Expected: `14 passed`

- [ ] **Step 5: Replace the old config test module**

`tests/test_config.py` tests module-level constants that no longer exist. Delete it — its coverage is superseded by `tests/test_config_file.py`:

```bash
git rm tests/test_config.py
```

- [ ] **Step 6: Write the failing test for config-driven tiers in worker**

Append to `tests/test_worker.py`:

```python
def test_main_builds_escalation_tiers_from_config(monkeypatch, tmp_path):
    """worker.main must turn the configured tier list into the
    (threshold, name, chat_fn) triples run_job already consumes."""
    from handyman import config as config_module

    cfg = config_module.Config(
        tiers=[
            config_module.Tier(name="small", model="m-small", threshold_tokens=0),
            config_module.Tier(name="mid", model="m-mid", threshold_tokens=24000),
        ],
        ollama_host="http://localhost:11434",
        max_concurrent_jobs=3,
        max_iterations=40,
        max_wall_clock_seconds=1200,
        watchdog_max_retries=3,
        tavily_api_key=None,
        db_path=tmp_path / "jobs.db",
        jobs_log_dir=tmp_path / "jobs",
    )
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(worker.config, "load", lambda *a, **k: cfg)

    conn = db.connect(cfg.db_path)
    job_id = db.create_job(conn, "task", str(tmp_path))
    conn.close()

    captured = {}

    def fake_run_job(*args, **kwargs):
        captured["escalation_tiers"] = kwargs["escalation_tiers"]
        captured["chat_fn"] = kwargs["chat_fn"]

    monkeypatch.setattr(worker, "run_job", fake_run_job)
    monkeypatch.setattr(worker.ollama_client, "model_is_pulled", lambda host, model: True)
    monkeypatch.setattr(worker, "spawn_worker", lambda job_id: None)

    worker.main(job_id)

    tiers = captured["escalation_tiers"]
    assert [(t[0], t[1]) for t in tiers] == [(24000, "mid")]
    assert callable(captured["chat_fn"])
```

- [ ] **Step 7: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_worker.py::test_main_builds_escalation_tiers_from_config -q`
Expected: FAIL — `AttributeError` on `worker.config.load`

- [ ] **Step 8: Rewrite worker.main to build tiers from config**

In `handyman/worker.py`, delete `_default_chat_fn`, `_mid_chat_fn`, and `_big_chat_fn` entirely, and replace `main` with:

```python
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
                conn,
                job_id,
                "error",
                result_summary=(
                    "no model tiers configured - run `handyman setup` to create "
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
```

- [ ] **Step 9: Thread the config through execute_tool_call and spawn_worker**

`execute_tool_call` currently reads `config.TAVILY_API_KEY`, which no longer exists as a module constant. Change its signature in `handyman/worker.py`:

```python
def execute_tool_call(working_dir: str, name: str, arguments: dict, cfg) -> str:
```

and inside it, replace `config.TAVILY_API_KEY` with `cfg.tavily_api_key`.

`spawn_worker` also references `config.JOBS_LOG_DIR`. Change it to load config:

```python
def spawn_worker(job_id: str) -> None:
    cfg = config.load()
    cfg.jobs_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.jobs_log_dir / f"{job_id}.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            # -m, not a file path: executing the file directly would put
            # handyman/ on sys.path instead of the repo root, breaking the
            # package-absolute imports at the top of this module.
            [sys.executable, "-m", "handyman.worker", job_id],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **procutil.detached_kwargs(),
        )
```

- [ ] **Step 9b: Update execute_tool_call's six test callers**

Step 9 added a fourth parameter to `execute_tool_call`, which breaks every existing caller in `tests/test_worker_dispatch.py`. Find them:

```bash
grep -n "execute_tool_call" tests/test_worker_dispatch.py
```

Each call needs a `Config` passed as the fourth argument. Add this helper at the top of `tests/test_worker_dispatch.py` and pass `_cfg()` to every call:

```python
from pathlib import Path

from handyman import config as config_module


def _cfg(tavily_api_key=None, **overrides):
    """Minimal Config for tool-dispatch tests, which only read tavily_api_key."""
    defaults = dict(
        tiers=[config_module.Tier(name="small", model="m", threshold_tokens=0)],
        ollama_host="http://localhost:11434",
        max_concurrent_jobs=3,
        max_iterations=40,
        max_wall_clock_seconds=1200,
        watchdog_max_retries=3,
        tavily_api_key=tavily_api_key,
        db_path=Path("unused.db"),
        jobs_log_dir=Path("unused-jobs"),
    )
    defaults.update(overrides)
    return config_module.Config(**defaults)
```

`test_execute_tool_call_web_search_passes_configured_tavily_key` currently monkeypatches a module constant; rewrite it to pass `_cfg(tavily_api_key="tvly-test")` and assert the key reaches `tools.web_search`.

- [ ] **Step 10: Update server.py's config references**

Find them:

```bash
grep -n "config\." handyman/server.py
```

Each of `config.DB_PATH`, `config.MAX_CONCURRENT_JOBS`, `config.JOBS_LOG_DIR` becomes a field on a loaded config. Add `cfg = config.load()` as the first line of each of `gemma_delegate`, `gemma_check`, and `gemma_cancel`, then use `cfg.db_path`, `cfg.max_concurrent_jobs`, `cfg.jobs_log_dir`.

- [ ] **Step 11: Update any remaining test references to old constants**

```bash
grep -rn "config\.\(DB_PATH\|JOBS_LOG_DIR\|MODEL_NAME\|MAX_\|OLLAMA_HOST\|TAVILY_API_KEY\|WATCHDOG_\|CONTEXT_GROWTH\)" tests/ handyman/
```

Expected after fixes: no output. Tests that monkeypatched module constants should instead monkeypatch `config.load` to return a `Config` built like the one in Step 6.

- [ ] **Step 12: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass. Count will be roughly `146` (136 from Task 2, minus 5 deleted `test_config.py` tests, plus 14 config-file tests, plus 1 worker tier test).

- [ ] **Step 13: Write a working config file and smoke-test a real job**

The tool now needs a config file to run at all. Create one matching this machine's current setup:

```bash
mkdir -p ~/.config/handyman
cat > ~/.config/handyman/config.yaml <<'EOF'
ollama_host: http://localhost:11434
max_concurrent_jobs: 3
tiers:
  - name: small
    model: gemma-12b-gpu:latest
    threshold_tokens: 0
  - name: mid
    model: gemma-12b-gpu-mid:latest
    threshold_tokens: 24000
  - name: big
    model: gemma-12b-gpu-big:latest
    threshold_tokens: 48000
EOF
```

With Ollama running, delegate a real job and confirm it completes:

```bash
.venv/Scripts/python.exe -m handyman.cli delegate "write a file called hello.txt containing the word hello" "$(pwd)/.superpowers/sdd/smoke-test-scratch/config-file"
```

Then poll with `.venv/Scripts/python.exe -m handyman.cli check <job_id>` until status is `done`, and verify `hello.txt` exists with the right content. Mocked tests do not count as verification for this project — a real job must complete.

- [ ] **Step 14: Commit**

```bash
git add -A
git commit -m "Replace hardcoded model constants with a YAML config file

Tier count is now data: 1-3 tiers, validated for ascending thresholds
and for a first tier named 'small' (run_job initializes current_tier to
db.BASE_TIER, so a differently-named first tier would silently break
cross-job tier blocking).

Config resolution is defaults < file < env. jobs.db and logs move to a
platform data dir, overridable via HANDYMAN_DB_PATH / HANDYMAN_DATA_DIR.
MAX_TOTAL_TOKENS deleted - dead since the token cap left run_job's
signature in Task 7 of the original build."
```

---

### Task 4: CI matrix across Windows, macOS, and Linux

Until this runs, "cross-platform" is a claim. Task 2's POSIX kill path has never executed on a POSIX machine.

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `pyproject.toml` from Task 1 (for `pip install -e .[dev]`).
- Produces: a green matrix on three OSes; required before Plans B and C.

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  test:
    name: ${{ matrix.os }} / py${{ matrix.python-version }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install
        run: python -m pip install --upgrade pip && python -m pip install -e ".[dev]"

      - name: Run tests
        run: python -m pytest -q
```

`fail-fast: false` is deliberate: when a POSIX-only bug appears, the Windows job's result is still wanted in the same run.

- [ ] **Step 2: Verify the workflow file parses**

```bash
.venv/Scripts/python.exe -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "Add CI matrix across Windows, macOS, and Linux

Task 2's POSIX process-tree kill has never run on a POSIX machine.
fail-fast is off so one platform's failure doesn't hide the others."
git push
```

Note: this repo has had no remote so far. If `git push` fails with "no upstream", the matrix cannot run — stop and ask the user whether to create a GitHub remote, since a CI-less "cross-platform" claim is exactly what this task exists to prevent.

- [ ] **Step 4: Watch the run and fix what the other platforms reveal**

Open the Actions tab, or run `gh run watch` if the GitHub CLI is available (it is deliberately not installed on this machine, so the web UI is expected).

Failures to anticipate, with their fixes:

- **`test_run_bash_timeout_kills_the_grandchild_promptly` fails or hangs on Linux/macOS** — `process_group_kwargs()` is not reaching the `Popen` call, so `os.getpgid` targets the wrong group. Verify `run_bash` passes `**procutil.process_group_kwargs()`.
- **Path assertions fail on POSIX** — tests comparing string paths with backslashes. Compare `Path` objects, not strings.
- **`tests/test_tools_bash.py` shell-syntax failures** — commands written for `cmd.exe` (`type`, `dir`, `echo %VAR%`) do not exist under `/bin/sh`. Rewrite the affected commands using `sys.executable -c` so they are shell-agnostic, matching the pattern already used in the timeout test.
- **`XDG_CONFIG_HOME` test fails on Windows** — it monkeypatches `os.name` to `"posix"` but `Path("/tmp/xdg")` is a `WindowsPath`. If this appears, mark that single assertion with `@pytest.mark.skipif(os.name == "nt", reason="POSIX path semantics")`.

Fix each on a branch, push, and confirm all six matrix cells go green.

- [ ] **Step 5: Commit the fixes**

```bash
git add -A
git commit -m "Fix cross-platform test failures surfaced by the CI matrix"
git push
```

- [ ] **Step 6: Confirm the matrix is green**

All six cells (3 OSes × 2 Python versions) must pass. Record the run URL in `.superpowers/sdd/progress.md`. Do not claim cross-platform support until this is green — that is the whole point of the task.

---

## Plans B and C (not yet written)

Deliberately deferred. Both consume interfaces this plan defines, and writing them now would mean inventing signatures Task 3 is meant to pin down. Write each as its own plan once its dependency lands.

**Plan B — Setup wizard.** Depends on Task 3's `config.Config`/`Tier` schema and `default_config_path()`. Covers: OS/GPU/VRAM detection with fixtures per platform; the curated candidate list from the spec's "Model selection policy"; the live tool-call self-test built from the four recorded response shapes in the spec's testing plan; the guided-question fallback when detection is ambiguous; writing the config file; and `handyman setup` as a CLI subcommand.

**Plan C — Hosted provider.** Depends on Task 3's config schema and Plan B's self-test (reused to verify a hosted model before writing it into config). Covers: the `provider` column and its idempotent migration; the provider decision table from the spec's "Data flow"; `allow_hosted` on `gemma_delegate`; hosted jobs bypassing the concurrency cap and tier bookkeeping; an OpenAI-compatible client reusing `ollama_client.chat`'s request shape; and terminal-error handling for the 16K input cap and hosted failures.

## Open items inherited from the spec

Neither blocks this plan:

1. **The name is a placeholder.** `handyman` collides with an established Java job scheduler and must be replaced before any public release. Global Constraints above keep the blast radius small.
2. **Official (non-abliterated) Gemma tool-calling is unverified locally.** Relevant to Plan B's curated list, not to this plan.
3. **Tavily `web_search` has never been called against the real API.** It is implemented and unit-tested with mocks only. The spec's testing plan lists this as an outstanding live-verification gap. It is unrelated to the foundation work, so it is not a task here — but it should be closed before release, and Task 3 Step 9 touches the code path that carries the key.
