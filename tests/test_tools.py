"""Tests for the tools module (execution, error strings, batch handler)."""

import json

from uharness import tools


def test_execute_tool_unknown():
    assert tools.execute_tool("nope", {}) == "error: unknown tool nope"


def test_execute_tool_error_is_string(tmp_path):
    result = tools.execute_tool("read_file", {"path": str(tmp_path / "missing.txt")})
    assert result.startswith("error: FileNotFoundError:")


def _call(name, args, call_id="id-1"):
    return [{"id": call_id, "function": {"name": name, "arguments": args}}]


def test_handler_returns_tool_messages_and_calls(tmp_path):
    (tmp_path / "a.txt").write_text("data")
    args = json.dumps({"path": str(tmp_path / "a.txt")})
    results, _calls = tools.handler(_call("read_file", args))
    assert results == [
        {
            "role": "tool",
            "tool_call_id": "id-1",
            "content": "data",
        }
    ]


def test_handler_malformed_arguments():
    results, calls = tools.handler(_call("read_file", "{not valid json"))
    assert len(results) == 1
    assert calls == [{"tool": "read_file", "path": None}]
    assert results[0]["tool_call_id"] == "id-1"
    assert results[0]["content"].startswith("error: malformed tool arguments")


def test_handler_none_arguments(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    args = json.dumps({"path": str(empty)})
    results, calls = tools.handler(
        [
            {"id": "a", "function": {"name": "list_dir", "arguments": args}},
            {"id": "b", "function": {"name": "read_file", "arguments": None}},
        ]
    )
    assert calls == [
        {"tool": "list_dir", "path": str(empty)},
        {"tool": "read_file", "path": None},
    ]
    assert [r["tool_call_id"] for r in results] == ["a", "b"]
    assert all(r["role"] == "tool" for r in results)
    assert results[0]["content"] == ""
    assert results[1]["content"].startswith("error: TypeError:")
