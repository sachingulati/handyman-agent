"""
Every OS-specific branch lives here so the branches can be tested by patching
os.name in one place.
"""
import os
import shutil
import signal
import subprocess

def is_pid_alive(pid):
    """Best-effort liveness check; a dead worker must be reapable."""
    if pid is None or pid == 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        # Default restype is c_int, which truncates a 64-bit HANDLE.
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            # GetExitCodeProcess(hProcess, lpExitCode) takes a POINTER to a
            # DWORD that receives the exit code - it does not return it.
            # Calling it with one argument makes ctypes pass an arbitrary
            # address as the out-pointer, and Windows writes through it.
            #
            # OpenProcess can also succeed for a process that has already
            # exited but whose object is not yet torn down, so the exit code
            # is the only reliable liveness signal.
            exit_code = ctypes.c_ulong(0)
            ok = kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        else:
            return True

def process_group_kwargs():
    """Passes arguments to define a start for a new session."""
    if os.name == "nt":
        return {}
    return {"start_new_session": True}

def detached_kwargs():
    """Defines flags used when starting an orphaned/detached process."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}

def kill_process_tree(pid):
    """Kill a process and everything it spawned.

    process.kill() only kills the immediate child. With shell=True that
    child is cmd.exe or /bin/sh, and the real command runs as a grandchild
    that survives - leaving communicate() blocked until the orphan exits.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                        capture_output=True, text=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def shell_command(command: str):
    """Turn a shell string into (argv_or_string, use_shell).

    `shell=True` picks the platform shell, which on Windows is cmd.exe.
    That is not a dialect difference to shrug at: single quotes do not
    quote, `|` inside them is still a pipe, and an executable path with
    forward slashes is not found. Commands written as ordinary POSIX shell
    fail there in ways that read as model incompetence rather than a
    platform mismatch.

    So prefer a real POSIX shell when the machine has one - Git for
    Windows ships bash - and fall back to the platform default only when
    it does not.
    """
    for candidate in ("bash", "sh"):
        found = shutil.which(candidate)
        if found:
            return [found, "-c", command], False
    return command, True
