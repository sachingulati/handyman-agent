import os
import pytest
from pathlib import Path

from handyman import tools
def test_write_then_read_file(tmp_path):
    tools.write_file(str(tmp_path), "note.txt", "hello world")
    assert tools.read_file(str(tmp_path), "note.txt") == "hello world"


def test_write_file_creates_parent_dirs(tmp_path):
    tools.write_file(str(tmp_path), "sub/dir/note.txt", "nested")
    assert (tmp_path / "sub" / "dir" / "note.txt").read_text() == "nested"


def test_read_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tools.read_file(str(tmp_path), "missing.txt")


def test_edit_file_replaces_unique_match(tmp_path):
    tools.write_file(str(tmp_path), "note.txt", "the quick fox")
    tools.edit_file(str(tmp_path), "note.txt", "quick", "slow")
    assert tools.read_file(str(tmp_path), "note.txt") == "the slow fox"


def test_edit_file_raises_if_not_found(tmp_path):
    tools.write_file(str(tmp_path), "note.txt", "the quick fox")
    with pytest.raises(ValueError, match="not found"):
        tools.edit_file(str(tmp_path), "note.txt", "missing", "x")


def test_edit_file_raises_if_not_unique(tmp_path):
    tools.write_file(str(tmp_path), "note.txt", "fox fox fox")
    with pytest.raises(ValueError, match="not unique"):
        tools.edit_file(str(tmp_path), "note.txt", "fox", "cat")


def test_path_escape_via_dotdot_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), "../outside.txt", "escape")
    assert not outside.exists()


def test_path_escape_via_absolute_path_is_rejected(tmp_path):
    other = tmp_path.parent / "other_dir"
    other.mkdir()
    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), str(other / "f.txt"), "escape")


def test_resolve_in_jail_allows_nested_path(tmp_path):
    resolved = tools.resolve_in_jail(str(tmp_path), "a/b/c.txt")
    assert resolved == (tmp_path / "a" / "b" / "c.txt").resolve()


# Windows-specific jail-escape regression tests (Task 3 review)


def test_path_escape_via_midpath_dotdot_is_rejected(tmp_path):
    """Escape via mid-path `..`: 'sub/../../escape.txt' traverses back out after going one level in."""
    # Create a subdirectory to make the traversal pattern realistic
    (tmp_path / "sub").mkdir()
    outside = tmp_path.parent / "escape.txt"

    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), "sub/../../escape.txt", "escape")

    assert not outside.exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows-only escape vector; on POSIX these strings name a "
           "file inside the jail rather than escaping it",
)
def test_path_escape_via_backslash_dotdot_is_rejected(tmp_path):
    """Escape via backslash-style traversal: '..\\..\\escape.txt'."""
    outside = tmp_path.parent.parent / "escape.txt"

    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), "..\\..\\escape.txt", "escape")

    assert not outside.exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows-only escape vector; on POSIX these strings name a "
           "file inside the jail rather than escaping it",
)
def test_path_escape_via_mixed_slash_traversal_is_rejected(tmp_path):
    """Escape via mixed forward/backslash traversal: 'sub\\../../escape.txt'."""
    (tmp_path / "sub").mkdir()
    outside = tmp_path.parent / "escape.txt"

    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), "sub\\../../escape.txt", "escape")

    assert not outside.exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows-only escape vector; on POSIX these strings name a "
           "file inside the jail rather than escaping it",
)
def test_path_escape_via_different_drive_is_rejected(tmp_path):
    """Escape via different-drive absolute path (e.g., 'D:\\escape.txt' when tmp_path is on C:)."""
    tmp_drive = Path(str(tmp_path)).drive

    # Pick a different drive letter than tmp_path's drive
    if tmp_drive.upper() == "C:":
        escape_path = "D:\\escape.txt"
    else:
        escape_path = "C:\\escape.txt"

    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), escape_path, "escape")

    # Verify nothing was written to the target drive
    target = Path(escape_path)
    assert not target.exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows-only escape vector; on POSIX these strings name a "
           "file inside the jail rather than escaping it",
)
def test_path_escape_via_unc_path_is_rejected(tmp_path):
    """Escape via UNC path: '\\\\server\\share\\escape.txt'."""
    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), "\\\\server\\share\\escape.txt", "escape")


def test_path_escape_via_directory_junction_is_rejected(tmp_path):
    """Escape via directory-junction symlink pointing outside tmp_path."""
    import subprocess
    import os

    # Create a target directory outside tmp_path
    outside_dir = tmp_path.parent / "outside_junction_target"
    outside_dir.mkdir(exist_ok=True)

    # Create a junction inside tmp_path pointing to the outside directory
    junction_path = tmp_path / "junction"

    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_path), str(outside_dir)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        pytest.skip(f"Directory junction creation not feasible in this environment: {e}")

    # Attempt to write through a direct path into the junction
    escape_file = outside_dir / "escape.txt"

    with pytest.raises(tools.PathJailViolation):
        tools.write_file(str(tmp_path), "junction/escape.txt", "escape")

    assert not escape_file.exists()

    # Cleanup
    try:
        junction_path.unlink()
    except Exception:
        pass
    try:
        outside_dir.rmdir()
    except Exception:
        pass
