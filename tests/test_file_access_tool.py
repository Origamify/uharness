"""Tests for the file_access_tool module."""

import os

import pytest

from uharness import file_access_tool, path_permissions


def test_read_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\nworld")
    assert file_access_tool.read_file(str(f)) == "hello\nworld"


def test_read_file_too_large(tmp_path):
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (file_access_tool.MAX_READ_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        file_access_tool.read_file(str(f))


def test_read_numbered_pages_past_the_size_cap(tmp_path):
    # whole file exceeds the cap → read_file refuses, but a small range deep in it works
    lines = [f"line-{i:05d}" for i in range(1, 20001)]
    f = tmp_path / "big.txt"
    f.write_text("\n".join(lines) + "\n")
    assert os.path.getsize(f) > file_access_tool.MAX_READ_BYTES

    assert (
        file_access_tool.read_numbered(str(f), 19000, 19002)
        == "19000: line-19000\n19001: line-19001\n19002: line-19002"
    )


def test_read_numbered_range_obeying_the_size_cap(tmp_path):
    # unbounded range on a too-large file → truncated at the cap with a notice
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line-{i:05d}" for i in range(1, 20001)) + "\n")

    result = file_access_tool.read_numbered(str(f))

    lines = result.split("\n")
    assert lines[0] == "1: line-00001"
    assert len(result.encode()) <= file_access_tool.MAX_READ_BYTES
    notice = lines[-1]
    last_fetched = int(notice.split("fetched lines 1-")[1].split(";")[0])
    assert notice.startswith(
        f"[{file_access_tool.MAX_READ_BYTES} byte read limit reached; fetched lines 1-"
    )
    assert notice.endswith(f"continue with start_line={last_fetched + 1}]")
    assert lines[last_fetched - 1] == f"{last_fetched}: line-{last_fetched:05d}"


def test_read_numbered_single_line_over_cap_gets_notice(tmp_path):
    f = tmp_path / "huge_line.txt"
    f.write_text("x" * (file_access_tool.MAX_READ_BYTES + 1) + "\n")
    result = file_access_tool.read_numbered(str(f), 1, 1)
    assert result == (
        f"[{file_access_tool.MAX_READ_BYTES} byte read limit reached; "
        "no single line at 1 fits the limit]"
    )


def test_read_numbered_numbers_the_whole_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    assert file_access_tool.read_numbered(str(f)) == "1: alpha\n2: beta\n3: gamma"

    # no trailing newline → still one line; empty file → empty output
    f.write_text("only")
    assert file_access_tool.read_numbered(str(f)) == "1: only"
    (tmp_path / "empty.txt").write_text("")
    assert file_access_tool.read_numbered(str(tmp_path / "empty.txt")) == ""


def test_read_numbered_range_clamps_to_file_bounds(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("l1\nl2\nl3\nl4\nl5\n")

    # 1-based inclusive endpoints; line numbers keep their file positions
    assert file_access_tool.read_numbered(str(f), 2, 4) == "2: l2\n3: l3\n4: l4"
    # omitted end → to EOF; ends past EOF clamp; start past EOF → empty
    assert file_access_tool.read_numbered(str(f), 4) == "4: l4\n5: l5"
    assert file_access_tool.read_numbered(str(f), 4, 99) == "4: l4\n5: l5"
    assert file_access_tool.read_numbered(str(f), 7, 9) == ""


def test_write_file(tmp_path):
    f = tmp_path / "out.txt"
    f.write_text("old")
    result = file_access_tool.write_file(str(f), "abc")
    assert f.read_text() == "abc"
    assert result == f"wrote 3 chars to {f}"


def test_write_file_creates_missing_dirs(tmp_path):
    f = tmp_path / "a" / "b" / "c.txt"
    result = file_access_tool.write_file(str(f), "abc")
    assert f.read_text() == "abc"
    assert result == f"wrote 3 chars to {f}"


def test_edit_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("foo\nbar\nbaz")
    result = file_access_tool.edit_file(str(f), "bar", "BAR")
    assert f.read_text() == "foo\nBAR\nbaz"
    assert result == f"edited {f}"


@pytest.mark.parametrize(
    "content, old_text, match",
    [
        ("hello", "nope", "not found"),
        ("x\ny\nx", "x", "ambiguous"),
    ],
)
def test_edit_file_rejects_bad_match(tmp_path, content, old_text, match):
    f = tmp_path / "a.txt"
    f.write_text(content)
    with pytest.raises(ValueError, match=match):
        file_access_tool.edit_file(str(f), old_text, "z")


def test_expanduser(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "a.txt").write_text("~content")
    assert file_access_tool.read_file("~/a.txt") == "~content"
    result = file_access_tool.write_file("~/b.txt", "x")
    assert (tmp_path / "b.txt").read_text() == "x"
    assert result == "wrote 1 chars to ~/b.txt"
    (tmp_path / "c.txt").write_text("old")
    file_access_tool.edit_file("~/c.txt", "old", "new")
    assert (tmp_path / "c.txt").read_text() == "new"


def test_delete_path_moves_file_to_trash(tmp_path):
    f = tmp_path / "gone.txt"
    f.write_text("bye")
    result = file_access_tool.delete_path(str(f))
    assert result == f"moved {f} to {tmp_path / '.trash'}"
    assert not f.exists()
    assert (tmp_path / ".trash" / "gone.txt").read_text() == "bye"


def test_delete_path_moves_dir_to_trash(tmp_path):
    d = tmp_path / "gone_dir"
    (d / "inner").mkdir(parents=True)
    (d / "inner" / "f.txt").write_text("x")
    file_access_tool.delete_path(str(d))
    assert not d.exists()
    assert (tmp_path / ".trash" / "gone_dir" / "inner" / "f.txt").read_text() == "x"


def test_delete_path_custom_trash(tmp_path):
    trash = tmp_path / "bin"
    f = tmp_path / "gone.txt"
    f.write_text("bye")
    file_access_tool.delete_path(str(f), trash=str(trash))
    assert (trash / "gone.txt").read_text() == "bye"
    assert not (tmp_path / ".trash").exists()


def test_delete_path_overwrites_same_named_trash_entry(tmp_path):
    f = tmp_path / "gone.txt"
    f.write_text("new")
    (tmp_path / ".trash").mkdir()
    (tmp_path / ".trash" / "gone.txt").write_text("old")

    file_access_tool.delete_path(str(f))

    assert (tmp_path / ".trash" / "gone.txt").read_text() == "new"


def test_move_path(tmp_path):
    src = tmp_path / "a" / "f.txt"
    src.parent.mkdir()
    src.write_text("data")
    dest = tmp_path / "b" / "c" / "g.txt"
    result = file_access_tool.move_path(str(src), str(dest))
    assert result == f"moved {src} to {dest}"
    assert not src.exists()
    assert dest.read_text() == "data"


def test_move_path_dir(tmp_path):
    src = tmp_path / "a" / "dir"
    (src / "inner").mkdir(parents=True)
    (src / "inner" / "f.txt").write_text("x")
    dest = tmp_path / "b" / "c" / "moved_dir"
    file_access_tool.move_path(str(src), str(dest))
    assert (dest / "inner" / "f.txt").read_text() == "x"


def test_move_path_overwrites_dest(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("new")
    dest = tmp_path / "dest.txt"
    dest.write_text("old")
    file_access_tool.move_path(str(src), str(dest))
    assert dest.read_text() == "new"


def test_list_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_text("")
    names = file_access_tool.list_dir(str(tmp_path)).splitlines()
    assert names == ["file.txt", "sub/"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert file_access_tool.list_dir(str(empty)) == ""


def test_denied_path_rejected(tmp_path):
    path_permissions.set_allowed_roots([])
    f = tmp_path / "a.txt"
    f.write_text("secret")
    with pytest.raises(PermissionError):
        file_access_tool.read_file(str(f))


def test_schemas_match_tool_map():
    schemas = {s["function"]["name"]: s for s in file_access_tool.FILE_ACCESS_TOOL}
    assert set(schemas) == set(file_access_tool._FILE_TOOL_MAP)
    for schema in file_access_tool.FILE_ACCESS_TOOL:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["parameters"]["type"] == "object"
        assert isinstance(fn["description"], str) and fn["description"]
