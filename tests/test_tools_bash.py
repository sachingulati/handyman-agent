import sys
import time

from handyman import tools
def test_run_bash_captures_stdout(tmp_path):
    cmd = f'"{sys.executable}" -c "print(1 + 1)"'
    result = tools.run_bash(str(tmp_path), cmd)
    assert result["stdout"].strip() == "2"
    assert result["return_code"] == 0


def test_run_bash_runs_in_working_dir(tmp_path):
    (tmp_path / "marker.txt").write_text("here")
    cmd = f'"{sys.executable}" -c "import os; print(os.path.exists(\'marker.txt\'))"'
    result = tools.run_bash(str(tmp_path), cmd)
    assert result["stdout"].strip() == "True"


def test_run_bash_captures_nonzero_exit(tmp_path):
    cmd = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    result = tools.run_bash(str(tmp_path), cmd)
    assert result["return_code"] == 3


def test_run_bash_captures_stderr(tmp_path):
    cmd = f'"{sys.executable}" -c "import sys; sys.stderr.write(\'oops\')"'
    result = tools.run_bash(str(tmp_path), cmd)
    assert "oops" in result["stderr"]


def test_run_bash_handles_timeout(tmp_path):
    # Command that sleeps for 3 seconds and only writes a marker file
    # AFTER the sleep completes, with a run_bash timeout of 1 second.
    # If the whole process tree (not just the cmd.exe shell wrapper) is
    # genuinely killed, the marker must never appear. If only the shell
    # wrapper were killed, the orphaned python process would keep running
    # in the background and write the marker ~3s after it started.
    marker = tmp_path / "marker.txt"
    cmd = (
        f'"{sys.executable}" -c "import time; time.sleep(3); '
        f'open(r\'{marker}\', \'w\').write(\'done\')"'
    )

    start = time.monotonic()
    result = tools.run_bash(str(tmp_path), cmd, timeout=1)
    elapsed = time.monotonic() - start

    # Should not raise an exception
    assert isinstance(result, dict)
    # Should have the expected keys
    assert "stdout" in result
    assert "stderr" in result
    assert "return_code" in result
    # Return code should be -1 for timeout
    assert result["return_code"] == -1
    # Timeout message should be in stderr
    assert "timed out" in result["stderr"]

    # The call must return close to the configured timeout, not block for
    # the full 3s sleep duration of the runaway grandchild process.
    assert elapsed < 2.5, f"run_bash took {elapsed:.2f}s, expected close to the 1s timeout"

    # Give the orphaned process the time it would have needed to wake up
    # from its sleep and write the marker, if it had survived the timeout.
    time.sleep(3)
    assert not marker.exists(), "orphaned grandchild process was not killed on timeout"


def test_run_bash_handles_posix_quoting_and_pipes(tmp_path):
    """The exact construction that failed on Windows: single quotes with a
    pipe inside. Under cmd.exe the pipe was treated as a pipeline and the
    command died with "'b' is not recognized"."""
    result = tools.run_bash(str(tmp_path), "echo 'a|b|c'")
    assert result["return_code"] == 0
    assert "a|b|c" in result["stdout"]


def test_run_bash_expands_a_glob(tmp_path):
    (tmp_path / "one.txt").write_text("x", encoding="utf-8")
    (tmp_path / "two.txt").write_text("y", encoding="utf-8")
    result = tools.run_bash(str(tmp_path), "ls *.txt")
    assert result["return_code"] == 0
    assert "one.txt" in result["stdout"] and "two.txt" in result["stdout"]
