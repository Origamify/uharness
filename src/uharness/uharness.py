"""Harness core: the configurable job runner with the tool-calling loop."""

from . import api_call, file_access_tool, tools
from .path_permissions import PathPermissions
from .result_writer import ResultWriter

MAX_ROUNDS = 20


class AbortRun(RuntimeError):
    """run() aborted by the caller's stop callback; the document stays partial."""


# Tools exposed when no allowlist is set: read, write, edit, and list.
DEFAULT_TOOLS = frozenset({"read_file", "write_file", "edit_file", "list_dir"})


class Uharness:
    """Configurable job runner: env/model/rounds fixed at init, permissions and tool allowlist settable."""

    def __init__(
        self, env_path=api_call.DEFAULT_ENV_PATH, model=None, base_url=None,
        api_key=None, path_permissions=(), max_rounds=MAX_ROUNDS,
    ):
        self.env_path = env_path
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_rounds = max_rounds
        self._allowed_tools = DEFAULT_TOOLS
        self._pp = PathPermissions(path_permissions)
        self._tool_map = file_access_tool.make_tool_map(self._pp)
        self._extra_schemas = []
        self._extra_tool_map = {}

    def register_tool(self, schema, fn):
        """Add a custom tool for this harness: OpenAI function schema + callable.

        The callable receives the call's arguments as keyword arguments and
        returns a string; its exceptions reach the model as error strings, like
        built-in tools. The tool is automatically added to the allowlist. A
        malformed schema raises TypeError (wrong types) or ValueError (bad
        values); duplicate names raise ValueError; a non-callable fn raises
        TypeError.
        """
        tools._validate_tool_schema(schema)
        name = schema["function"]["name"]
        if name in self._tool_map or name in self._extra_tool_map:
            raise ValueError(f"tool {name!r} already registered")
        if not callable(fn):
            raise TypeError(f"tool {name!r}: fn is not callable")
        self._extra_schemas.append(schema)
        self._extra_tool_map[name] = fn
        self._allowed_tools = self._allowed_tools | {name}

    def set_path_permissions(self, paths):
        """Replace this harness's allowed-root list; file tools validate against it at call time."""
        self._pp.set_roots(paths)

    def set_trash_path(self, path):
        """Set where delete_path moves entries; None restores the per-directory .trash default."""
        self._tool_map = file_access_tool.make_tool_map(self._pp, trash=path)

    def set_allowed_tools(self, names):
        """Tool names the model may see; None (default) allows read/write/edit/list."""
        self._allowed_tools = DEFAULT_TOOLS if names is None else frozenset(names)

    def list_tools(self, show_all=False):
        """Sorted tool names visible to the model; show_all ignores the allowlist."""
        available = [
            t["function"]["name"] for t in [*tools.TOOLS, *self._extra_schemas]
        ]
        if not show_all:
            available = [n for n in available if n in self._allowed_tools]
        return sorted(available)

    def run(
        self, prompt, reasoning="medium", output_path=None, tool_schemas=None, stop=None
    ):
        """One job: loop model ↔ tools until final answer, return the full result document.

        With output_path, the document is flushed to disk after every event so
        partial results are always readable. Without it, nothing is written.
        ``stop``, when given, is a zero-argument callable checked before every
        model round; a truthy return aborts the job by raising AbortRun (the
        partial document stays readable). Raises RuntimeError if the model
        doesn't produce a final answer within max_rounds; other errors raise —
        caller decides how to surface them.
        """
        if tool_schemas is None:
            tool_schemas = [
                t
                for t in [*tools.TOOLS, *self._extra_schemas]
                if t["function"]["name"] in self._allowed_tools
            ]
        writer = ResultWriter(output_path, prompt=prompt)
        messages = [{"role": "user", "content": prompt}]
        for _ in range(self.max_rounds):
            if stop is not None and stop():
                raise AbortRun("aborted by caller")
            message, usage = api_call.chat(
                messages,
                tools=tool_schemas,
                model=self.model,
                reasoning=reasoning,
                env_path=self.env_path,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            writer.add_tokens(usage.get("total_tokens", 0))
            messages.append(message)
            writer.add_assistant(message)
            if not message.get("tool_calls"):
                writer.finish(message["content"] or "", len(prompt))
                return writer.document()
            tool_messages, calls = tools.handler(
                message["tool_calls"],
                tool_map={**self._tool_map, **self._extra_tool_map},
            )
            for call in calls:
                writer.console_line(f"$ {call['tool']} {call['path'] or ''}".rstrip())
                writer.tool_call(call)
            messages.extend(tool_messages)
        raise RuntimeError(f"No final answer after {self.max_rounds} rounds")
