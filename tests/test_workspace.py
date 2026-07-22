import os
import pathlib
import subprocess
import sys

from handyman import workspace


def test_clear_bytecode_removes_caches(tmp_path):
    """Stale bytecode once produced three convincing false test failures."""
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"x")
    assert workspace.clear_bytecode(tmp_path) == 1
    assert not (tmp_path / "pkg" / "__pycache__").exists()


def test_clear_bytecode_is_safe_when_there_is_none(tmp_path):
    assert workspace.clear_bytecode(tmp_path) == 0


def test_check_syntax_reports_a_broken_file(tmp_path):
    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text('y = f"unterminated\n', encoding="utf-8")
    broken = workspace.check_syntax(tmp_path)
    assert len(broken) == 1
    assert "bad.py" in broken[0]


def test_check_syntax_ignores_venv_and_caches(tmp_path):
    venv = tmp_path / ".venv" / "Lib"
    venv.mkdir(parents=True)
    (venv / "broken.py").write_text("def (", encoding="utf-8")
    assert workspace.check_syntax(tmp_path) == []


def test_lint_flags_shell_mangled_escapes():
    problems = workspace.lint_brief('Write \\"\\"\\"a docstring\\"\\"\\" then stop.')
    assert any("backslash" in p for p in problems)


def test_lint_flags_a_missing_finish_line():
    problems = workspace.lint_brief("Write handyman/thing.py with four functions.")
    assert any("finish line" in p for p in problems)


def test_lint_passes_a_good_brief():
    assert workspace.lint_brief(
        "Write handyman/thing.py so tests/test_thing.py passes. "
        "One write_file call, then stop."
    ) == []


def test_reset_recreates_a_dirty_workspace(tmp_path):
    target = tmp_path / "ws"
    (target / "handyman").mkdir(parents=True)
    (target / "handyman" / "stale.py").write_text("old", encoding="utf-8")
    tests = tmp_path / "src"
    tests.mkdir()
    (tests / "test_a.py").write_text("def test_a(): pass\n", encoding="utf-8")

    workspace.reset(target, tests_from=tests, deps=())

    assert not (target / "handyman" / "stale.py").exists()
    assert (target / "tests" / "test_a.py").exists()


def test_reset_raises_rather_than_silently_leaving_files(tmp_path, monkeypatch):
    """Deleting can report success while leaving files behind if something
    holds them open. Silently continuing means a later run inherits stale
    modules, which is worse than failing."""
    target = tmp_path / "ws"
    target.mkdir()
    monkeypatch.setattr(workspace.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(workspace.time, "sleep", lambda s: None)
    try:
        workspace.reset(target, deps=())
    except RuntimeError as exc:
        assert "could not be removed" in str(exc)
    else:
        raise AssertionError("reset should refuse to continue")


def test_processes_touching_finds_a_live_process():
    marker = "handyman-workspace-probe"
    proc = subprocess.Popen([sys.executable, "-c", f"import time;#{marker}\ntime.sleep(5)"])
    try:
        found = workspace.processes_touching(marker)
        assert any(p["ProcessId"] == proc.pid for p in found)
    finally:
        proc.kill()
        proc.wait()


def test_kill_processes_reports_survivors_not_just_targets():
    """"I killed it" has been wrong often enough that the caller is told
    what actually survived."""
    targets, survivors = workspace.kill_processes("handyman-nothing-matches-this")
    assert targets == []
    assert survivors == []
