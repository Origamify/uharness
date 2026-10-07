"""Tests for the harness core (no live server needed)."""

import json

import pytest

from uharness import AbortRun, ResultWriter, Uharness, tools
from uharness import uharness as harness

# A two-round conversation: one tool call, then a final answer.
ASSISTANT_TOOL = {
    "role": "assistant",
    "reasoning_content": "think",
    "tool_calls": [
        {
            "id": "1",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "a.txt"}),
            },
        }
    ],
}
FINAL = {"role": "assistant", "content": "done"}

EXPECTED = {
    "console": "$ read_file a.txt",
    "output": "done",
    # The user prompt is intentionally NOT an event.
    "events": [
        "reasoning> think",
        'tool> read_file {"path": "a.txt"}',
        "content> done",
    ],
    "tool_calls": [{"tool": "read_file", "path": "a.txt"}],
    "prompt": "hi",
    "prompt_length": 2,
    "total_tokens": 8,
}


def _stub_rounds(monkeypatch):
    """Stub chat + handler to replay a fixed two-round conversation."""
    seq = [(ASSISTANT_TOOL, {"total_tokens": 5}), (FINAL, {"total_tokens": 3})]
    monkeypatch.setattr(
        harness.api_call,
        "chat",
        lambda messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False: (
            seq.pop(0)
        ),
    )
    monkeypatch.setattr(
        harness.tools,
        "handler",
        lambda tcs, tool_map=None: (
            [{"role": "tool", "tool_call_id": "1", "content": "..."}],
            [{"tool": "read_file", "path": "a.txt"}],
        ),
    )


def test_run_error_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(harness.api_call, "chat", boom)
    with pytest.raises(RuntimeError, match="api down"):
        Uharness().run("hi")


def test_run_stop_before_first_round(monkeypatch):
    """A stop already truthy aborts before any model call."""

    def unreachable(*args, **kwargs):
        raise AssertionError("chat must not be called")

    monkeypatch.setattr(harness.api_call, "chat", unreachable)

    with pytest.raises(AbortRun):
        Uharness().run("hi", stop=lambda *_: True)


def test_run_stop_between_rounds(tmp_path, monkeypatch):
    """Stop flips after round one: round two never starts, partial stays readable."""
    out = tmp_path / "result.json"
    state = {"stop": False}

    def chat_then_stop(
        messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False,
    ):
        state["stop"] = True  # flips after round one's model call
        return (ASSISTANT_TOOL, {"total_tokens": 5})

    monkeypatch.setattr(harness.api_call, "chat", chat_then_stop)
    monkeypatch.setattr(
        harness.tools,
        "handler",
        lambda tcs, tool_map=None: (
            [{"role": "tool", "tool_call_id": "1", "content": "..."}],
            [{"tool": "read_file", "path": "a.txt"}],
        ),
    )

    with pytest.raises(AbortRun):
        Uharness().run("hi", output_path=str(out), stop=lambda *_: state["stop"])

    assert 'tool> read_file {"path": "a.txt"}' in json.loads(out.read_text())["events"]


def test_run_no_final_answer_raises(monkeypatch):
    monkeypatch.setattr(
        harness.api_call,
        "chat",
        lambda messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False: (
            (ASSISTANT_TOOL, {"total_tokens": 1})
        ),
    )

    with pytest.raises(RuntimeError, match="No final answer after 3 rounds"):
        Uharness(max_rounds=3).run("hi")


def test_run_succeeds_on_final_round(monkeypatch):
    rounds = [(ASSISTANT_TOOL, {"total_tokens": 1})] * 2 + [
        (FINAL, {"total_tokens": 1})
    ]
    monkeypatch.setattr(
        harness.api_call,
        "chat",
        lambda messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False: (
            rounds.pop(0)
        ),
    )
    monkeypatch.setattr(
        harness.tools,
        "handler",
        lambda tcs, tool_map=None: (
            [{"role": "tool", "tool_call_id": "1", "content": "..."}],
            [{"tool": "read_file", "path": "a.txt"}],
        ),
    )

    assert Uharness(max_rounds=3).run("hi")["output"] == "done"


def test_run_uses_default_max_rounds(monkeypatch):
    calls = {"n": 0}

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        calls["n"] += 1
        return (ASSISTANT_TOOL, {"total_tokens": 1})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    monkeypatch.setattr(
        harness.tools,
        "handler",
        lambda tcs, tool_map=None: (
            [{"role": "tool", "tool_call_id": "1", "content": "..."}],
            [{"tool": "read_file", "path": "a.txt"}],
        ),
    )

    with pytest.raises(RuntimeError, match="No final answer after 20 rounds"):
        Uharness().run("hi")
    assert calls["n"] == 20


def test_run_empty_final_content_yields_empty_output(monkeypatch):
    monkeypatch.setattr(
        harness.api_call,
        "chat",
        lambda messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False: (
            {"role": "assistant", "content": None},
            {"total_tokens": 1},
        ),
    )

    doc = Uharness().run("hi")

    assert doc["output"] == ""
    assert doc["events"] == []


def test_run_returns_document(monkeypatch):
    _stub_rounds(monkeypatch)
    doc = Uharness().run("hi")

    assert doc["execution_time"] >= 0
    del doc["execution_time"]
    assert doc == EXPECTED


def test_run_writes_result_file(tmp_path, monkeypatch):
    _stub_rounds(monkeypatch)
    out = tmp_path / "result.json"

    doc = Uharness().run("hi", output_path=str(out))

    assert json.loads(out.read_text()) == doc


def test_result_writer_creates_parent_dirs(tmp_path, monkeypatch):
    _stub_rounds(monkeypatch)
    out = tmp_path / "nested" / "dir" / "result.json"

    Uharness().run("hi", output_path=str(out))

    assert json.loads(out.read_text())["output"] == "done"


def test_incremental_writes_during_run(tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    _stub_rounds(monkeypatch)

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        if len(messages) == 1:  # first round
            return (ASSISTANT_TOOL, {"total_tokens": 5})
        # Second round: the file must already hold round one's events.
        partial = json.loads(out.read_text())
        assert 'tool> read_file {"path": "a.txt"}' in partial["events"]
        return (FINAL, {"total_tokens": 3})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)

    Uharness().run("hi", output_path=str(out))


def test_run_without_output_path_writes_nothing(tmp_path, monkeypatch):
    _stub_rounds(monkeypatch)
    Uharness().run("hi")
    assert list(tmp_path.iterdir()) == []


def test_allowed_tools_restrict_schemas(monkeypatch):
    captured = {}

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        captured["schemas"] = tools
        return (FINAL, {"total_tokens": 1})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    h = Uharness()
    h.set_allowed_tools({"read_file", "list_dir"})
    h.run("hi")

    names = {t["function"]["name"] for t in captured["schemas"]}
    assert names == {"read_file", "list_dir"}

    # None resets to the default four (an earlier custom allowlist is not remembered)
    h.set_allowed_tools({"move_path"})
    h.set_allowed_tools(None)
    h.run("hi")

    names = {t["function"]["name"] for t in captured["schemas"]}
    assert names == {"read_file", "write_file", "edit_file", "list_dir"}


def test_list_tools_defaults_and_allowlist():
    h = Uharness()

    assert h.list_tools() == ["edit_file", "list_dir", "read_file", "write_file"]
    h.set_allowed_tools({"read_file", "list_dir"})
    assert h.list_tools() == ["list_dir", "read_file"]


def test_list_tools_show_all_includes_opt_in_and_custom():
    h = Uharness()
    h.set_allowed_tools({"read_file"})
    h.register_tool(_custom_tool_schema(), lambda city: "x")

    assert h.list_tools() == ["get_weather", "read_file"]
    assert h.list_tools(show_all=True) == [
        "delete_path",
        "edit_file",
        "get_weather",
        "list_dir",
        "move_path",
        "read_file",
        "read_numbered",
        "write_file",
    ]


def test_set_trash_path(monkeypatch, tmp_path):
    src = tmp_path / "f.txt"
    src.write_text("x")
    trash = tmp_path / "bin"
    h = Uharness(path_permissions=[tmp_path])
    h.set_trash_path(str(trash))
    results, _ = tools.handler(
        [
            {
                "id": "1",
                "function": {
                    "name": "delete_path",
                    "arguments": json.dumps({"path": str(src)}),
                },
            }
        ],
        tool_map=h._tool_map,
    )
    assert "error" not in results[0]["content"]
    assert (trash / "f.txt").read_text() == "x"


def _custom_tool_schema(name="get_weather"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "custom tool",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }


def test_register_tool_end_to_end(monkeypatch):
    """Registered schema reaches the model; its result flows back as a tool message."""
    calls = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "oslo"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    seen = []

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        seen.append((list(tools or ()), list(messages)))
        return (calls.pop(0), {"total_tokens": 1})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    h = Uharness()
    h.register_tool(_custom_tool_schema(), lambda city: f"snow in {city}")
    h.set_allowed_tools(
        {"get_weather", "read_file", "write_file", "edit_file", "list_dir"}
    )
    doc = h.run("hi")

    schema_names = {t["function"]["name"] for t in seen[0][0]}
    assert "get_weather" in schema_names
    tool_msg = next(m for m in seen[1][1] if m["role"] == "tool")
    assert tool_msg["content"] == "snow in oslo"
    assert doc["tool_calls"] == [{"tool": "get_weather", "path": None}]


def test_register_tool_allowlist_interactions(monkeypatch):
    """Registered tools are auto-allowed; an explicit allowlist replaces that (None drops them)."""
    captured = {}

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        captured["schemas"] = tools
        return ({"role": "assistant", "content": "done"}, {"total_tokens": 1})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    h = Uharness()
    h.register_tool(_custom_tool_schema(), lambda city: "x")

    # auto-allowed on registration
    assert "get_weather" in h.list_tools()
    h.run("hi")
    assert "get_weather" in {t["function"]["name"] for t in captured["schemas"]}

    # explicit allowlist keeps exactly what it lists
    h.set_allowed_tools({"get_weather"})
    h.run("hi")
    assert {t["function"]["name"] for t in captured["schemas"]} == {"get_weather"}

    # None restores the default four — re-list registered tools there to keep them
    h.set_allowed_tools(None)
    assert h.list_tools() == ["edit_file", "list_dir", "read_file", "write_file"]


def test_register_tool_error_string_reaches_model(monkeypatch):
    calls = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "oslo"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    seen = []

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        seen.append(list(messages))
        return (calls.pop(0), {"total_tokens": 1})

    def boom(city):
        raise KeyError("no such city")

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    h = Uharness()
    h.register_tool(_custom_tool_schema(), boom)
    doc = h.run("hi")

    assert doc["output"] == "done"
    tool_msg = next(m for m in seen[1] if m["role"] == "tool")
    assert tool_msg["content"] == "error: KeyError: 'no such city'"


@pytest.mark.parametrize(
    ("bad_schema", "exc"),
    [
        ("not a dict", TypeError),
        ({"type": "function"}, TypeError),  # function missing
        ({"type": "function", "function": {}}, TypeError),  # name missing
        (
            {"type": "function", "function": {"name": "ok", "description": 3}},
            TypeError,
        ),
        (
            {"type": "function", "function": {"name": "ok", "parameters": []}},
            TypeError,
        ),
        (
            {
                "type": "function",
                "function": {"name": "ok", "parameters": {"properties": []}},
            },
            TypeError,
        ),
        ({"type": "tool", "function": {"name": "ok"}}, ValueError),
        ({"type": "function", "function": {"name": ""}}, ValueError),
        ({"type": "function", "function": {"name": "has space"}}, ValueError),
        ({"type": "function", "function": {"name": "x" * 65}}, ValueError),
    ],
)
def test_register_tool_rejects_bad_schemas(bad_schema, exc):
    with pytest.raises((TypeError, ValueError), match="schema"):
        Uharness().register_tool(bad_schema, lambda **kw: "x")


def test_register_tool_accepts_minimal_valid_schema():
    h = Uharness()
    h.register_tool(
        {
            "type": "function",
            "function": {
                "name": "ping_1",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        lambda *_: "pong",
    )

    assert "ping_1" in h._extra_tool_map


def test_register_tool_rejects_duplicates_and_non_callables():
    h = Uharness()
    h.register_tool(_custom_tool_schema(), lambda city: "x")

    with pytest.raises(ValueError, match="already registered"):
        h.register_tool(_custom_tool_schema(), lambda city: "y")
    with pytest.raises(ValueError, match="already registered"):
        h.register_tool(_custom_tool_schema("read_file"), lambda path: "y")
    with pytest.raises(TypeError, match="not callable"):
        h.register_tool(_custom_tool_schema("other"), "nope")


def test_registered_tools_are_per_instance(monkeypatch):
    captured = {}

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        captured["schemas"] = tools
        return ({"role": "assistant", "content": "done"}, {"total_tokens": 1})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    a = Uharness()
    a.register_tool(_custom_tool_schema(), lambda city: "x")
    Uharness().run("hi")

    names = {t["function"]["name"] for t in captured["schemas"]}
    assert "get_weather" not in names


def test_explicit_tool_schemas_override_defaults(monkeypatch):
    captured = {}
    schemas = [{"type": "function", "function": {"name": "custom", "parameters": {}}}]

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        captured["schemas"] = tools
        return (FINAL, {"total_tokens": 1})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    Uharness().run("hi", tool_schemas=schemas)

    assert captured["schemas"] is schemas


def test_config_kwargs_passed_through(monkeypatch):
    captured = {}

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        captured.update(
            reasoning=reasoning, env_path=env_path, model=model,
            base_url=base_url, api_key=api_key, stream=stream,
        )
        return (FINAL, {"total_tokens": 1})

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    Uharness(
        env_path="/my/app.env", model="detail", base_url="http://x", api_key="k"
    ).run("hi", reasoning="low")

    assert captured == {
        "reasoning": "low",
        "env_path": "/my/app.env",
        "model": "detail",
        "base_url": "http://x",
        "api_key": "k",
        "stream": False,
    }


def test_run_records_calls_and_tokens(monkeypatch):
    calls = [
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "function": {"name": "list_dir", "arguments": "{}"}}
                ],
            },
            {"total_tokens": 5},
        ),
        ({"role": "assistant", "content": "done"}, {"total_tokens": 3}),
    ]
    monkeypatch.setattr(
        harness.api_call,
        "chat",
        lambda messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False: (
            calls.pop(0)
        ),
    )

    result = Uharness().run("hi")

    assert result["output"] == "done"
    assert result["tool_calls"] == [{"tool": "list_dir", "path": None}]
    assert result["console"] == "$ list_dir"
    assert result["prompt"] == "hi"
    assert result["prompt_length"] == 2
    assert result["total_tokens"] == 8
    assert result["execution_time"] >= 0
    assert result["events"] == ["tool> list_dir {}", "content> done"]


def test_run_survives_malformed_tool_arguments(monkeypatch):
    bad_call = {
        "role": "assistant",
        "tool_calls": [
            {"id": "1", "function": {"name": "read_file", "arguments": "not json"}}
        ],
    }
    rounds = [
        (bad_call, {"total_tokens": 5}),
        ({"role": "assistant", "content": "done"}, {"total_tokens": 3}),
    ]
    seen = []

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        seen.append(list(messages))  # snapshot: run() keeps mutating the list
        return rounds.pop(0)

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)

    doc = Uharness().run("hi")

    assert doc["output"] == "done"
    assert doc["tool_calls"] == [{"tool": "read_file", "path": None}]
    assert doc["events"][0] == "tool> read_file not json"
    # the malformed arguments came back to the model as a tool message, not a crash
    assert seen[1][-1]["content"].startswith("error: malformed tool arguments")


def test_run_honors_path_permissions(tmp_path, monkeypatch):
    """Constructor roots allow reads inside and deny outside, through run()."""
    inside = tmp_path / "in"
    inside.mkdir()
    (inside / "a.txt").write_text("data")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    calls = [
        {
            "id": "1",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(inside / "a.txt")}),
            },
        },
        {
            "id": "2",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(outside)}),
            },
        },
    ]
    rounds = [
        ({"role": "assistant", "tool_calls": calls}, {"total_tokens": 2}),
        ({"role": "assistant", "content": "done"}, {"total_tokens": 1}),
    ]
    seen = []

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        seen.append(list(messages))
        return rounds.pop(0)

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)

    doc = Uharness(path_permissions=[str(inside)]).run("hi")

    assert doc["output"] == "done"
    results = [m for m in seen[1] if m["role"] == "tool"]
    assert results[0]["content"] == "data"
    assert results[1]["content"].startswith("error: PermissionError")


def test_set_path_permissions_replaces_roots(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    (a / "f.txt").write_text("A")
    b = tmp_path / "b"
    b.mkdir()
    (b / "g.txt").write_text("B")
    calls = [
        {
            "id": "1",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(a / "f.txt")}),
            },
        },
        {
            "id": "2",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(b / "g.txt")}),
            },
        },
    ]
    rounds = [
        ({"role": "assistant", "tool_calls": calls}, {"total_tokens": 2}),
        ({"role": "assistant", "content": "done"}, {"total_tokens": 1}),
    ]
    seen = []

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        seen.append(list(messages))
        return rounds.pop(0)

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)

    h = Uharness(path_permissions=[str(a)])
    h.set_path_permissions([str(b)])
    doc = h.run("hi")

    assert doc["output"] == "done"
    results = [m for m in seen[1] if m["role"] == "tool"]
    # old root revoked, new root works
    assert results[0]["content"].startswith("error: PermissionError")
    assert results[1]["content"] == "B"


# The same two-round conversation as SSE chunk lists (deltas split by hand).
STREAM_ROUNDS = [
    [
        {"choices": [{"delta": {"reasoning_content": "th"}}]},
        {"choices": [{"delta": {"reasoning_content": "ink"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "1", "function": {"name": "read_file", "arguments": "{\"pat"}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "h\": \"a.txt\"}"}}
        ]}}]},
        {"choices": [{"delta": {}}], "usage": None},
        {"choices": [], "usage": {"total_tokens": 5}},
    ],
    [
        {"choices": [{"delta": {"content": "do"}}]},
        {"choices": [{"delta": {"content": "ne"}}]},
        {"choices": [], "usage": {"total_tokens": 3}},
    ],
]


def test_run_stream_returns_same_document(monkeypatch):
    """Stream mode assembles deltas into the exact document a plain run yields."""
    rounds = [list(r) for r in STREAM_ROUNDS]

    def fake_chat(messages, tools=None, model=None, reasoning="high", env_path="~/.env", base_url=None, api_key=None, stream=False):
        assert stream is True
        return rounds.pop(0)

    monkeypatch.setattr(harness.api_call, "chat", fake_chat)
    monkeypatch.setattr(
        harness.tools,
        "handler",
        lambda tcs, tool_map=None: (
            [{"role": "tool", "tool_call_id": "1", "content": "..."}],
            [{"tool": "read_file", "path": "a.txt"}],
        ),
    )

    doc = Uharness().run("hi", stream=True)

    assert doc["execution_time"] >= 0
    del doc["execution_time"]
    assert doc == EXPECTED


def test_run_stream_flushes_events_live(tmp_path, monkeypatch):
    """The output file holds valid JSON mid-stream, events growing chunk by chunk."""
    out = tmp_path / "result.json"
    seen = []

    def gen():
        yield {"choices": [{"delta": {"content": "he"}}]}
        seen.append(json.loads(out.read_text())["events"])
        yield {"choices": [{"delta": {"content": "llo"}}]}
        yield {"choices": [], "usage": {"total_tokens": 1}}

    monkeypatch.setattr(harness.api_call, "chat", lambda *a, **k: gen())

    doc = Uharness().run("hi", output_path=str(out), stream=True)

    assert seen == [["content> he"]]  # readable and already grown after chunk one
    assert doc["output"] == "hello"
    assert doc["events"] == ["content> hello"]


def test_result_writer_without_path_only_builds_document():
    writer = ResultWriter(prompt="hi")
    writer.event("content> done")
    writer.console_line("$ read_file a.txt")
    writer.add_tokens(4)
    writer.finish("done", 2)

    doc = writer.document()
    assert doc["events"] == ["content> done"]
    assert doc["console"] == "$ read_file a.txt"
    assert doc["output"] == "done"
    assert doc["prompt_length"] == 2
    assert doc["total_tokens"] == 4
