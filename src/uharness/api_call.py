"""API call layer: one POST to an OpenAI-compatible chat completions endpoint.

The endpoint is pure configuration, supplied by the caller: an env file with
URL/KEY/model slots, or direct base_url/api_key kwargs. Nothing provider-
specific lives here — see .env.example at the project root.
"""

import json
import os
import urllib.request

DEFAULT_ENV_PATH = "~/.env"
TIMEOUT = 2400

# Accepted reasoning= values; "off" omits the field from the payload.
REASONING_LEVELS = frozenset({"off", "low", "medium", "high"})

MODEL_SLOTS = {"detail": "DETAIL_MODEL", "fast": "FAST_MODEL"}

# expanded path → env vars; loaded lazily on first API call, once per process
_ENV_CACHE: dict = {}


def _load_env(path: str) -> dict:
    values = {}
    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def _env(path: str) -> dict:
    """Env-file vars for this path; a missing file reads as empty."""
    key = os.path.expanduser(path)
    if key not in _ENV_CACHE:
        try:
            _ENV_CACHE[key] = _load_env(path)
        except FileNotFoundError:
            _ENV_CACHE[key] = {}
    return _ENV_CACHE[key]


def _resolve_model(model, env: dict):
    """Slot name → model string from the env file; raw model names are rejected."""
    if model in (None, "default"):
        return env.get("MODEL")
    if model in MODEL_SLOTS:
        return env.get(MODEL_SLOTS[model]) or env.get("MODEL")
    raise ValueError(
        f"model must be None, 'default', 'detail', or 'fast', got {model!r}"
    )


def chat(
    messages: list,
    tools: list | None = None,
    model: str | None = None,
    reasoning: str = "medium",
    env_path: str = DEFAULT_ENV_PATH,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Send messages (and optional tools) to the endpoint; return (assistant message, usage).

    The env file supplies the endpoint config: URL (required), KEY (optional —
    keyless servers omit it), and the model slots MODEL / DETAIL_MODEL /
    FAST_MODEL ("detail"/"fast" fall back to MODEL). base_url and api_key
    override URL and KEY directly — programmatic config without a file.
    """
    env = _env(env_path)
    url = base_url or env.get("URL")
    if not url:
        raise RuntimeError(f"{env_path} is missing URL")
    if reasoning not in REASONING_LEVELS:
        raise ValueError(
            f"reasoning must be one of {sorted(REASONING_LEVELS)}, got {reasoning!r}"
        )
    payload = {"model": _resolve_model(model, env), "messages": messages}
    if reasoning != "off":
        payload["reasoning_effort"] = reasoning
    if tools:
        payload["tools"] = tools
    key = api_key or env.get("KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"], data.get("usage", {})
