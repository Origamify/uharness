"""Tool schemas and name->callable map, combined from tool modules."""

import json
import re

from . import file_access_tool

TOOLS = file_access_tool.FILE_ACCESS_TOOL
_TOOL_MAP = file_access_tool._FILE_TOOL_MAP

# OpenAI tool-name rule: 1-64 chars of a-z, A-Z, 0-9, "_" or "-".
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_tool_schema(schema):
    """Raise TypeError (wrong types) or ValueError (bad values) for a bad schema."""
    if not isinstance(schema, dict):
        raise TypeError("schema must be a dict")
    if schema.get("type") != "function":
        raise ValueError('schema["type"] must be "function"')
    function = schema.get("function")
    if not isinstance(function, dict):
        raise TypeError('schema["function"] must be a dict')
    name = function.get("name")
    if not isinstance(name, str):
        raise TypeError('schema["function"]["name"] must be a string')
    if not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            'schema["function"]["name"] must be 1-64 chars of a-z, A-Z, 0-9, "_" or "-"'
        )
    if not isinstance(function.get("description", ""), str):
        raise TypeError('schema["function"]["description"] must be a string')
    parameters = function.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError('schema["function"]["parameters"] must be a JSON Schema dict')
    if isinstance(parameters, dict) and not isinstance(
        parameters.get("properties", {}), dict
    ):
        raise TypeError('schema["function"]["parameters"]["properties"] must be a dict')


def execute_tool(name: str, arguments: dict, tool_map=None) -> str:
    """Run a tool by name; return its result or an error string the model can read."""
    fn = (tool_map or _TOOL_MAP).get(name)
    if fn is None:
        return f"error: unknown tool {name}"
    try:
        return fn(**arguments)
    except Exception as e:  # noqa: BLE001 — let the model see the failure and retry
        return f"error: {type(e).__name__}: {e}"


def handler(tool_calls: list, tool_map=None) -> tuple:
    """Execute each tool call; return (tool messages to append, call records)."""
    results = []
    calls = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"] or "{}"
        try:
            fn_args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError) as e:
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"error: malformed tool arguments {raw_args!r}: {e}",
                }
            )
            calls.append({"tool": name, "path": None})
            continue
        results.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": execute_tool(name, fn_args, tool_map),
            }
        )
        calls.append({"tool": name, "path": fn_args.get("path")})
    return results, calls
