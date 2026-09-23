"""Tests for the api_call module."""

import json

import pytest

from uharness import api_call


class FakeResponse:
    """Minimal stand-in for the urlopen context-manager response."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Patch urlopen; capture the Request and kwargs it receives.

    Also clears the env cache so tests never read the real ~/.env.
    """
    captured = {}
    monkeypatch.setattr(api_call, "_ENV_CACHE", {})

    def _fake(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        body = json.dumps(
            {
                "choices": [{"message": {"content": "hello from model"}}],
                "usage": {"total_tokens": 42},
            }
        ).encode()
        return FakeResponse(body)

    monkeypatch.setattr(api_call.urllib.request, "urlopen", _fake)
    return captured


@pytest.fixture
def env_file(tmp_path):
    """Write a tmp env file (a fresh path per call) and return its path."""
    counter = [0]

    def _write(content: str) -> str:
        counter[0] += 1
        path = tmp_path / f"app{counter[0]}.env"
        path.write_text(content)
        return str(path)

    return _write


def test_load_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "not an assignment\n"
        "  SPACED   =   padded value  \n"
        "URL=http://x?a=b\n",
    )
    assert api_call._load_env(str(env)) == {
        "SPACED": "padded value",
        "URL": "http://x?a=b",
    }


def test_chat_posts_env_config_and_returns_message(fake_urlopen, env_file):
    path = env_file("URL=http://testserver/v1/chat/completions\nKEY=test-key\nMODEL=test-model\n")

    message, usage = api_call.chat([{"role": "user", "content": "hi"}], env_path=path)
    assert message["content"] == "hello from model"
    assert usage == {"total_tokens": 42}

    req = fake_urlopen["req"]
    assert req.full_url == "http://testserver/v1/chat/completions"
    assert req.method == "POST"
    assert req.get_header("Authorization") == "Bearer test-key"
    payload = json.loads(req.data)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["reasoning_effort"] == "medium"  # default level
    assert fake_urlopen["timeout"] == api_call.TIMEOUT


def test_key_is_optional(fake_urlopen, env_file):
    """No KEY in the env file → no Authorization header."""
    path = env_file("URL=http://testserver/v1/chat/completions\nMODEL=m\n")

    api_call.chat([{"role": "user", "content": "hi"}], env_path=path)

    assert fake_urlopen["req"].get_header("Authorization") is None


def test_missing_url_raises(fake_urlopen, env_file):
    path = env_file("KEY=x\n")
    with pytest.raises(RuntimeError, match="is missing URL"):
        api_call.chat([{"role": "user", "content": "hi"}], env_path=path)


def test_missing_env_file_reads_as_empty(fake_urlopen):
    """A nonexistent env file behaves as empty: missing URL, no crash on open."""
    with pytest.raises(RuntimeError, match="is missing URL"):
        api_call.chat(
            [{"role": "user", "content": "hi"}],
            env_path="/nonexistent/does-not-exist.env",
        )


def test_env_is_cached_per_path(fake_urlopen, env_file):
    """The env file is read once per path; later edits don't leak into calls."""
    path = env_file("URL=http://first\nMODEL=first-model\n")

    api_call.chat([{"role": "user", "content": "hi"}], env_path=path)
    with open(path, "w") as f:  # edit after the first call
        f.write("URL=http://second\nMODEL=second-model\n")
    api_call.chat([{"role": "user", "content": "hi"}], env_path=path)

    assert fake_urlopen["req"].full_url == "http://first"
    assert json.loads(fake_urlopen["req"].data)["model"] == "first-model"


def test_env_cache_is_keyed_by_path(fake_urlopen, env_file):
    """Different env paths hold different configs in the same process."""
    a = env_file("URL=http://a\nMODEL=model-a\n")
    b = env_file("URL=http://b\nMODEL=model-b\n")

    api_call.chat([{"role": "user", "content": "hi"}], env_path=a)
    api_call.chat([{"role": "user", "content": "hi"}], env_path=b)

    assert fake_urlopen["req"].full_url == "http://b"
    assert json.loads(fake_urlopen["req"].data)["model"] == "model-b"


@pytest.mark.parametrize(
    ("model", "env", "expected"),
    [
        (None, "MODEL=default-model\n", "default-model"),
        ("default", "MODEL=default-model\n", "default-model"),
        ("detail", "MODEL=base\nDETAIL_MODEL=big-model\n", "big-model"),
        ("fast", "MODEL=base\nFAST_MODEL=quick-model\n", "quick-model"),
        ("detail", "MODEL=base\n", "base"),  # unset slot falls back to MODEL
        ("fast", "MODEL=base\n", "base"),
    ],
)
def test_model_slot_resolution(fake_urlopen, env_file, model, env, expected):
    path = env_file(f"URL=http://testserver\n{env}")

    api_call.chat([{"role": "user", "content": "hi"}], model=model, env_path=path)

    assert json.loads(fake_urlopen["req"].data)["model"] == expected


def test_raw_model_names_are_rejected(fake_urlopen, env_file):
    """model= selects a slot only — raw model strings live in the env file."""
    path = env_file("URL=http://testserver\nMODEL=m\n")
    for bad in ["some-vendor-model", "", "DETAIL", "Fast"]:
        with pytest.raises(ValueError, match="model must be"):
            api_call.chat(
                [{"role": "user", "content": "hi"}], model=bad, env_path=path
            )


def test_base_url_and_api_key_override_env(fake_urlopen, env_file):
    """Direct kwargs win over the env file's URL/KEY."""
    path = env_file("URL=http://from-env\nKEY=env-key\nMODEL=env-model\n")

    api_call.chat(
        [{"role": "user", "content": "hi"}],
        env_path=path,
        base_url="http://override",
        api_key="override-key",
    )

    req = fake_urlopen["req"]
    assert req.full_url == "http://override"
    assert req.get_header("Authorization") == "Bearer override-key"


def test_base_url_without_env_file(fake_urlopen):
    """Envless mode: base_url alone is enough; the file never has to exist."""
    api_call.chat(
        [{"role": "user", "content": "hi"}],
        env_path="/nonexistent/does-not-exist.env",
        base_url="http://llm.local:8000/v1/chat/completions",
    )
    assert fake_urlopen["req"].full_url == "http://llm.local:8000/v1/chat/completions"


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_reasoning_level_in_payload(fake_urlopen, env_file, level):
    path = env_file("URL=http://testserver\nMODEL=m\n")

    api_call.chat(
        [{"role": "user", "content": "hi"}], reasoning=level, env_path=path
    )

    assert json.loads(fake_urlopen["req"].data)["reasoning_effort"] == level


@pytest.mark.parametrize("level", ["xhigh", "max", "weird", ""])
def test_unsupported_reasoning_level_raises(fake_urlopen, env_file, level):
    """Levels the endpoint vocabulary doesn't define are rejected, not translated."""
    path = env_file("URL=http://testserver\nMODEL=m\n")

    with pytest.raises(ValueError, match="reasoning must be one of"):
        api_call.chat(
            [{"role": "user", "content": "hi"}], reasoning=level, env_path=path
        )


def test_reasoning_off_omits_field(fake_urlopen, env_file):
    path = env_file("URL=http://testserver\nMODEL=m\n")

    api_call.chat([{"role": "user", "content": "hi"}], reasoning="off", env_path=path)

    assert "reasoning_effort" not in json.loads(fake_urlopen["req"].data)


def test_tools_included_in_payload(fake_urlopen, env_file):
    path = env_file("URL=http://testserver\nMODEL=m\n")
    schema = [{"type": "function", "function": {"name": "ping", "parameters": {}}}]

    api_call.chat([{"role": "user", "content": "hi"}], tools=schema, env_path=path)
    assert json.loads(fake_urlopen["req"].data)["tools"] == schema

    # no tools → key omitted entirely
    api_call.chat([{"role": "user", "content": "hi"}], env_path=path)
    assert "tools" not in json.loads(fake_urlopen["req"].data)


def test_env_cache_hit_avoids_rereading(monkeypatch, tmp_path):
    """_env loads lazily and then serves from the cache (file read exactly once)."""
    monkeypatch.setattr(api_call, "_ENV_CACHE", {})
    path = str(tmp_path / "app.env")
    with open(path, "w") as f:
        f.write("URL=http://x\n")
    reads = []
    real_load = api_call._load_env

    def counting_load(p):
        reads.append(p)
        return real_load(p)

    monkeypatch.setattr(api_call, "_load_env", counting_load)

    api_call._env(path)
    api_call._env(path)

    assert reads == [str(path)]
    assert api_call._ENV_CACHE[str(path)] == {"URL": "http://x"}
