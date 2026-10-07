# uharness — Agent Orientation

`about.md` for purpose/usage, `interface.md` for the API + result contract.

## Status

Active. Hosts embed uharness in-process, one instance per job (no state between runs). No TODOs; nothing known broken.

## Scope

In:
- Importable library running one LLM job per `run()`: prompt → model ↔ tool loop (cap 20 rounds) → result dict, optionally flushed incrementally to JSON; `stream=True` consumes the response over SSE with events growing live
- Built-in file tools (read/write/edit/list; delete/move/numbered-read opt-in), path-permission gating, custom tool registration
- Any OpenAI-compatible chat-completions endpoint, configured entirely by the caller's env file (or direct URL/key kwargs) — no provider registry in code

Out of scope:
- General agent framework; CLI surface
- Job queueing/worker dispatch, task tracking, web/TUI surfaces — host-app concerns
- Benchmarking/evaluation

## Run & setup

Library — no launcher. Install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then `from uharness import Uharness` (see `about.md` / `interface.md`). Tests: `./test.sh` (offline; API call is faked, tmp-dir-only path permissions).

## File map

| Path | Role |
|---|---|
| `about.md` | Human intro: what/why, install, minimal usage |
| `interface.md` | Library contract: `Uharness`/`run()` parameters, result schema, error behavior |
| `pyproject.toml` | Package metadata; zero runtime dependencies (stdlib only) |
| `.env.example` | Template env file carrying all endpoint config (URL/KEY/model slots); copy and fill in per deployment |
| `test.sh` | Runs the pytest suite |
| `src/uharness/__init__.py` | Public API: exports `Uharness`, `ResultWriter`, `AbortRun`, `MAX_ROUNDS` |
| `src/uharness/uharness.py` | Core: `Uharness` (env path/model slot/max_rounds at init; path permissions, trash path, tool allowlist settable) and `run()` — the tool-calling loop with `stop=` abort and `stream=` SSE mode (`_stream_round` assembles deltas into a message while events grow live). **Change loop behavior → here** |
| `src/uharness/api_call.py` | One POST to an OpenAI-compatible `/chat/completions`, plain or SSE-streamed; resolves URL/KEY/model slots from an env file (`env_path=`), validates reasoning levels. **Change request/endpoint handling → here; providers stay in the env file** |
| `src/uharness/tools.py` | Tool routing: merges tool modules' schemas + call maps into `TOOLS`/`_TOOL_MAP`; `handler()` executes a batch, converting failures to error strings. **Register a built-in tool module → here** |
| `src/uharness/file_access_tool.py` | File tools + their schemas; every path goes through `path_permissions.validate_path`. **Change file-tool behavior → here** |
| `src/uharness/path_permissions.py` | Realpath-under-allowed-roots validation; no roots → everything denied |
| `src/uharness/result_writer.py` | Result document + incremental flush (valid JSON after every event/console line, and per SSE chunk in stream mode); event strings (`reasoning>`, `tool>`, `content>`), live streamed events |
| `tests/` | pytest suite, all modules, no network |

## Conventions

- Tool functions raise on failure; `tools` catches and returns an error string so the model can retry — a tool call never crashes a run
- Result file is written incrementally, never buffer-and-write at the end
- Timeouts are a caller concern (thread/subprocess) or cooperative `stop=`; uharness does not self-limit
- Tests monkeypatch `uharness.api_call` and grant tmp roots; offline only

## Settings

Endpoint config lives in an env file (default `~/.env`, per-app via `env_path=`; loaded on first API call, never at import; `env_prefix` removed). Generic names — see `.env.example` at the project root:

| Name | Meaning | Default |
|---|---|---|
| `URL` | Full chat-completions endpoint | none — required unless `base_url=` is passed |
| `KEY` | Bearer token | none — keyless servers omit it |
| `MODEL` | Default model (`model=None`/`"default"`) | none |
| `DETAIL_MODEL` | Heavier model (`model="detail"`) | falls back to `MODEL` |
| `FAST_MODEL` | Quick/cheap model (`model="fast"`) | falls back to `MODEL` |

`base_url`/`api_key` on `Uharness` override `URL`/`KEY` directly (envless mode); `model=` takes slot names only — raw model names live in the env file.

Module constants (change them in the source file):

| Constant | Meaning | Default |
|---|---|---|
| `api_call.TIMEOUT` | Socket timeout per API POST (seconds) | `2400` |
| `api_call.REASONING_LEVELS` | Accepted `reasoning=` values | `off`, `low`, `medium`, `high` — others raise `ValueError` |
| `uharness.MAX_ROUNDS` | Default cap on model↔tool rounds | `20` |
| `uharness.DEFAULT_TOOLS` | Tool allowlist when none is set | `read_file`, `write_file`, `edit_file`, `list_dir` |
| `file_access_tool` read cap | Max bytes per `read_file` call | 100 KB |
| Trash location for `delete_path` | Where soft-deleted entries go | `.trash` beside the entry, or path set via `set_trash_path` |

Init/run parameters (`env_path`, `model` slot, `max_rounds`, `reasoning`, `output_path`, `tool_schemas`, `stop`, …) with defaults: see `interface.md`.
