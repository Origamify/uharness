# µharness

Minimal agent harness for LLM jobs: prompt in → tool-calling loop → result dict out.
A thin Python library you import and call — not a framework, no CLI. Python 3.10+,
zero runtime dependencies.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The endpoint comes from an env file (default `~/.env`) — see `.env.example` at
the project root. Copy and fill it in:

```bash
cp .env.example ~/.env && chmod 600 ~/.env
```

## Usage

```python
from uharness import Uharness

uh = Uharness(path_permissions=["/home/me/work"])  # roots the file tools may touch
result = uh.run(prompt, output_path="result.json", reasoning="high")

print(result["output"])   # final answer
print(result["events"])   # full trace: reasoning>, tool>, content> lines
```

`run()` loops model ↔ tools (default cap 20 rounds) and returns the result
document: `output`, `console`, `events`, `tool_calls`, `prompt`, `prompt_length`,
`total_tokens`, `execution_time`. With `output_path` the document stays valid
JSON on disk after every event, so partial runs are readable. Every public
parameter (`env_path`, `model`, `stop=`, `tool_schemas`, …): see `interface.md`.

## Endpoint configuration

Any OpenAI-compatible chat-completions server. The endpoint is pure data in
your env file, never code — no provider registry; switching providers means
editing that one file. Generic names:

```bash
URL=https://api.example.com/v1/chat/completions  # full endpoint (required)
KEY=<bearer token>                               # optional — keyless servers omit it
MODEL=my-model                                   # default model
DETAIL_MODEL=my-bigger-model                     # model="detail" (falls back to MODEL)
FAST_MODEL=my-cheap-model                        # model="fast" (falls back to MODEL)
```

Parsed simply: `KEY=value` per line, `#` comments, values taken literally (no
quotes or `export`). Loaded lazily on first API call, once per process; missing
`URL` fails fast with `RuntimeError: <path> is missing URL`. `model=` selects a
slot (`None`/`"default"`, `"detail"`, `"fast"`) — raw model names are rejected;
they live in the env file, never in client code. `base_url=`/`api_key=` on
`Uharness` override `URL`/`KEY` directly for envless mode.

## Tools

Built-ins: `read_file`, `write_file`, `edit_file`, `list_dir` (on by default);
`delete_path`, `move_path`, `read_numbered` are opt-in. Every path is validated
against the allowed roots set by `path_permissions=` (or
`uh.set_path_permissions()`); with no roots, everything is denied.

```python
uh.set_allowed_tools({"read_file", "list_dir"})  # replace the allowlist
uh.register_tool(SCHEMA, fn)                     # custom tool: OpenAI schema + callable(**args) -> str
uh.list_tools()                                  # names the model may call
uh.set_trash_path(path)                          # where delete_path soft-deletes
```

Tool functions never crash a run: failures are returned to the model as error
strings so it can retry.

## Errors

Errors raise; results don't. Missing `URL`, API failure, or no final answer
within `max_rounds` → `RuntimeError`; a truthy `stop=` callback (checked before
every round) → `AbortRun`, leaving a readable partial result file. Tool errors
are the exception — they never raise.

## Tests

```bash
./test.sh    # pytest, offline (API call stubbed, tmp-dir-only path permissions)
```

## Dependencies

None at runtime — standard library only; `pip install -e .` pulls nothing
beyond the build backend (setuptools). Tests use
[pytest](https://pytest.org): `pip install pytest`.

Name: "µ" (mu) — a micro harness, like the Greek letter. The package is plain `uharness` (ASCII module name); µ is branding.

---


- Built with AI assistance (PI agent / uHarness) — code, tests, docs.
- Models used: Qwen / GLM.
- Code is mostly AI-written, human-reviewed. Human focus on modularization, clarity of purpose, and the contracts the AI implements against.
- Docs are mostly written autonomously by the agent and reviewed strictly.
- Tests are mostly AI-generated, then maintained and reviewed, with certain human requirements on what must be tested.
- Goal: deliver good solutions to the problem — oversized results for the effort — not to control every detail.
