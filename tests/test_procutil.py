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


def test_shell_command_uses_a_posix_shell_when_available(monkeypatch):
    """`shell=True` resolves to cmd.exe on Windows, where single quotes do
    not quote, `|` inside them is a pipe, and forward-slash executable
    paths fail. Three real commands died that way, each looking like a
    model error. Prefer a POSIX shell when one exists."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(procutil.shutil, "which", lambda name: r"C:\Git\bin\bash.exe"
                        if name == "bash" else None)
    argv, use_shell = procutil.shell_command("echo 'a|b'")
    assert use_shell is False
    assert argv[0].endswith("bash.exe")
    assert argv[1] == "-c"
    assert argv[2] == "echo 'a|b'"


def test_shell_command_falls_back_to_the_platform_shell(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(procutil.shutil, "which", lambda name: None)
    argv, use_shell = procutil.shell_command("dir")
    assert use_shell is True
    assert argv == "dir"


def test_shell_command_on_posix_uses_sh_directly(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(procutil.shutil, "which", lambda name: "/bin/bash"
                        if name == "bash" else None)
    argv, use_shell = procutil.shell_command("echo hi")
    assert use_shell is False
    assert argv == ["/bin/bash", "-c", "echo hi"]
