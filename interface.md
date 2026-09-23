# Interface

Library contract for callers — hosts embed uharness in-process, one harness instance per job.

## Usage

```python
from uharness import Uharness

uh = Uharness(
    env_path="~/.env",  # env file supplying URL / KEY / model slots (see .env.example)
    max_rounds=20,  # cap on model↔tool rounds
    path_permissions=["/home/me/work"],  # allowed roots for the file tools
    model=None,  # slot: None/"default", "detail", or "fast" — names live in the env file
    base_url=None,  # direct URL override (envless mode); api_key= overrides KEY too
)
uh.set_allowed_tools({"read_file", "list_dir"})  # tool names the agent may call; None (default) = read_file, write_file, edit_file, list_dir
uh.register_tool(SCHEMA, fn)  # custom tool: OpenAI schema + callable(**arguments) -> str
uh.list_tools()  # sorted names the model may call; list_tools(show_all=True) ignores the allowlist

result = uh.run(
    prompt,  # the assembled prompt: context header + task prompt. Required.
    output_path="result.json",  # default None: nothing written, use the returned dict
    reasoning="high",  # off, low, medium, high (default: medium); anything else raises ValueError
    tool_schemas=None,  # explicit OpenAI schema list; overrides the default file tools
    stop=None,  # zero-arg callable checked before every model round; truthy -> AbortRun
)
```

- `env_path` — env file supplying the endpoint config: `URL` (full chat-completions endpoint, required), `KEY` (bearer token, optional — keyless servers omit it), and the model slots `MODEL`, `DETAIL_MODEL`, `FAST_MODEL`. Generic names, one file per app/context — see `.env.example` at the project root. Loaded lazily on first API call, once per process; a missing file reads as empty.
- `model` — selects a slot in the env file: `None`/`"default"` → `MODEL`, `"detail"` → `DETAIL_MODEL` (falls back to `MODEL`), `"fast"` → `FAST_MODEL` (falls back to `MODEL`). Raw model names are rejected with `ValueError` — model names live in the env file, never in client code.
- `base_url` / `api_key` — direct overrides of `URL`/`KEY` for programmatic config without a file on disk (the env file is still consulted for model slots).
- `path_permissions` — allowed roots for the file tools (also settable later via `uh.set_path_permissions(paths)`); with no roots set, every path is denied.
- `tool_schemas` — explicit OpenAI schema list; overrides the default file tools (and makes `set_allowed_tools` moot).
- `stop` — cooperative cancellation: a zero-argument callable checked before every model round; a truthy return aborts the job by raising `AbortRun` (a `RuntimeError` subclass). With `output_path` the partial document stays readable.
- `register_tool(schema, fn)` — adds a custom tool to this harness instance (built-ins unaffected, per instance) and automatically allows it. `fn` receives the call's arguments as keyword arguments and returns a string; its exceptions reach the model as error strings like built-in tools. The schema is validated at registration: malformed schemas raise `TypeError` (wrong types) or `ValueError` (bad values); duplicate names raise `ValueError`. A later `set_allowed_tools(names)` replaces the allowlist, so re-list registered tools there to keep them visible. Default allowlist is read, write, edit, and list (`read_file`, `write_file`, `edit_file`, `list_dir`); pass `None` to restore it. `delete_path`, `move_path`, and `read_numbered` are opt-in.
- `list_tools(show_all=False)` — sorted tool names visible to the model (built-ins plus registered customs, filtered by the allowlist). `show_all=True` lists every possible tool on this instance, ignoring the allowlist.
- `set_trash_path(path)` — where `delete_path` moves files and directories (same-named entries already there are overwritten). `None` (default) restores a per-directory `.trash` folder beside each deleted entry. The trash dir is caller-configured and not restricted by `path_permissions`.
- To feed a prompt from a file, read it yourself: `prompt = open(path).read()`.

Lower-level: `Uharness.run()` is the loop itself plus the full document build; `ResultWriter` holds the incrementally-written document. `MAX_ROUNDS` is the default round cap. `reasoning` takes the endpoint's own vocabulary only — `off` omits the `reasoning_effort` field, `low`/`medium`/`high` pass through, anything else raises `ValueError` (`api_call.REASONING_LEVELS`).

Timeouts are a caller concern: bound the `uh.run` call yourself (thread, `asyncio.to_thread`, subprocess), or cancel cooperatively with `stop=`. uharness does not self-limit.

## Result document

`uh.run()` returns the dict that is (incrementally) written to `output_path`:

```
{
  "console": "$ write_file src/auth.py",
  "prompt": "Refactor the auth module...",
  "output": "Auth module refactored; all tests pass.",
  "events": [
    "user>\nRefactor the auth module...",
    "reasoning>\n...",
    "tool> write_file 'src/auth.py'",
    "content>\nAuth module refactored; all tests pass."
  ],
  "tool_calls": [
    {"tool": "write_file", "path": "src/auth.py"}
  ],
  "prompt_length": 4821,
  "total_tokens": 31207,
  "execution_time": 96.4
}
```

- `console` — console output from the run (command echoes, logs).
- `prompt` — the full prompt text fed in.
- `output` — the agent's final answer text.
- `events` — the run flattened to readable event strings in order: `reasoning>\n<text>` for extended thinking, `tool> <name> '<path>'` per tool call (path omitted when absent), `content>\n<text>` for assistant text.
- `tool_calls` — every tool call made, tool name + path argument only, no content.
- `prompt_length` — length of the prompt fed in (characters).
- `total_tokens` — total tokens generated across all API calls.
- `execution_time` — wall-clock seconds from start to finish.

The result document is the single source of truth for the run's outcome; with `output_path` it holds valid JSON after every event, so partial runs are always readable.

## Errors

Errors raise — a missing `URL` in the env file raises `RuntimeError` (`<path> is missing URL`); an API failure or no final answer within `max_rounds` raises `RuntimeError`; a truthy `stop` callback raises `AbortRun` (also a `RuntimeError`); the result file may be partial. Tool errors are never raised: they become strings returned to the model so it can retry.
