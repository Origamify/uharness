"""Tests for the path_permissions module."""

import json

import pytest

from uharness import file_access_tool, path_permissions, tools


def test_allowed(tmp_path):
    path_permissions.set_allowed_roots([str(tmp_path)])
    assert path_permissions.validate_path(str(tmp_path)) == str(tmp_path)
    assert path_permissions.validate_path(str(tmp_path / "a.txt")) == str(
        tmp_path / "a.txt"
    )


def test_denied(tmp_path):
    path_permissions.set_allowed_roots([str(tmp_path)])
    with pytest.raises(PermissionError):
        path_permissions.validate_path("/etc/passwd")
    # "/X_evil" must not match an allowed root "/X" via string prefix.
    sibling = tmp_path.parent / (tmp_path.name + "_evil")
    sibling.mkdir()
    with pytest.raises(PermissionError):
        path_permissions.validate_path(str(sibling / "f.txt"))
    # Empty allow-list denies everything, including the root itself.
    path_permissions.set_allowed_roots([])
    with pytest.raises(PermissionError):
        path_permissions.validate_path(str(tmp_path))


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    path_permissions.set_allowed_roots([str(tmp_path)])
    with pytest.raises(PermissionError):
        path_permissions.validate_path(str(link))


def test_permissions_are_per_instance(tmp_path):
    """Two PathPermissions instances never see each other's roots."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "f.txt").write_text("A")
    pp_a = path_permissions.PathPermissions([str(a)])
    pp_b = path_permissions.PathPermissions([str(tmp_path / "b")])
    call = [
        {
            "id": "1",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(a / "f.txt")}),
            },
        }
    ]

    ok, _ = tools.handler(call, tool_map=file_access_tool.make_tool_map(pp_a))
    denied, _ = tools.handler(call, tool_map=file_access_tool.make_tool_map(pp_b))
    assert ok[0]["content"] == "A"
    assert denied[0]["content"].startswith("error: PermissionError")
