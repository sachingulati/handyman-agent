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
