# µharness

Minimal agent harness for LLM jobs: prompt in → tool-calling loop → result dict out.
A thin Python library you import and call — not a framework, no CLI.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

```python
from uharness import Uharness

uh = Uharness(path_permissions=["/home/me/work"])  # roots the file tools may touch
result = uh.run(prompt, output_path="result.json", reasoning="high")
```

`run()` loops model ↔ tools (capped rounds) and returns the result document;
with `output_path` it stays valid JSON on disk after every event, so partial
runs are readable. Pass `stream=True` to consume the response over SSE —
the events grow live, chunk by chunk, in the same file.

Typical flow: point an env file at your OpenAI-compatible endpoint, grant the
file tools some allowed roots, hand in a prompt, read `result["output"]` and
`result["events"]`.

Parameters, tool allowlists, custom tools, and the result schema: see `interface.md`.

Name: "µ" (mu) — a micro harness, like the Greek letter. The package is plain `uharness` (ASCII module name); µ is branding.
