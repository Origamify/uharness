"""File access tools exposed to the model as OpenAI function schemas."""

import os
import shutil
from functools import partial

from . import path_permissions

MAX_READ_BYTES = 100 * 1024


def _resolve(path: str, pp) -> str:
    """Validate path with the bound pp; None means the module default (standalone use)."""
    return pp.validate(path) if pp else path_permissions.validate_path(path)


def read_file(path: str, *, pp=None) -> str:
    """Read a text file from disk and return its contents."""
    full_path = _resolve(path, pp)
    size = os.path.getsize(full_path)
    if size > MAX_READ_BYTES:
        raise ValueError(f"file too large ({size} bytes; limit {MAX_READ_BYTES})")
    with open(full_path) as f:
        return f.read()


def read_numbered(
    path: str, start_line: int | None = None, end_line: int | None = None, *, pp=None
) -> str:
    """Read a text file and return its contents with 1-based line numbers.

    Optional inclusive start_line/end_line range; omitted ends default to the
    file's bounds, out-of-range ends clamp to them.
    """
    body = read_file(path, pp=pp).removesuffix("\n")
    lines = body.split("\n") if body else []
    start = max(start_line or 1, 1)
    end = min(end_line or len(lines), len(lines))
    return "\n".join(
        f"{i}: {line}" for i, line in enumerate(lines[start - 1 : end], start=start)
    )


def write_file(path: str, content: str, *, pp=None) -> str:
    """Write text content to a file on disk. Creates missing parent directories."""
    full_path = _resolve(path, pp)
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


def edit_file(path: str, old_text: str, new_text: str, *, pp=None) -> str:
    """Replace the first occurrence of old_text with new_text in a file.

    old_text must match exactly and appear exactly once; otherwise an error is
    raised so the model can retry with a more specific match.
    """
    full_path = _resolve(path, pp)
    with open(full_path) as f:
        content = f.read()
    count = content.count(old_text)
    if count == 0:
        raise ValueError("old_text not found in file")
    if count > 1:
        raise ValueError(
            f"old_text is ambiguous ({count} matches); use a more specific match"
        )
    with open(full_path, "w") as f:
        f.write(content.replace(old_text, new_text))
    return f"edited {path}"


def delete_path(path: str, *, pp=None, trash=None) -> str:
    """Soft-delete a file or directory by moving it to the trash dir.

    The default trash is a .trash folder beside the entry; trash= overrides it
    (how set_trash_path binds a harness). Same-named entries already in the
    trash are overwritten.
    """
    full_path = _resolve(path, pp)
    trash_dir = trash or os.path.join(os.path.dirname(full_path) or ".", ".trash")
    os.makedirs(trash_dir, exist_ok=True)
    shutil.move(full_path, os.path.join(trash_dir, os.path.basename(full_path)))
    return f"moved {path} to {trash_dir}"


def move_path(src: str, dest: str, *, pp=None) -> str:
    """Move or rename a file or directory. Creates missing parent directories of dest."""
    src_path = _resolve(src, pp)
    dest_path = _resolve(dest, pp)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    shutil.move(src_path, dest_path)
    return f"moved {src} to {dest}"


def list_dir(path: str = ".", *, pp=None) -> str:
    """List files and subdirectories in a directory (dirs marked with /)."""
    base = _resolve(path, pp)
    names = sorted(os.listdir(base))
    return "\n".join(
        n + "/" if os.path.isdir(os.path.join(base, n)) else n for n in names
    )


_FILE_TOOL_MAP = {
    "read_file": read_file,
    "read_numbered": read_numbered,
    "write_file": write_file,
    "edit_file": edit_file,
    "delete_path": delete_path,
    "move_path": move_path,
    "list_dir": list_dir,
}


def make_tool_map(pp, trash=None):
    """Tool map (name -> callable) with every tool bound to a specific PathPermissions instance."""
    tool_map = {name: partial(fn, pp=pp) for name, fn in _FILE_TOOL_MAP.items()}
    tool_map["delete_path"] = partial(delete_path, pp=pp, trash=trash)
    return tool_map


FILE_ACCESS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_numbered",
            "description": "Read a text file and return its contents with 1-based line numbers. The optional start_line/end_line pair limits output to an inclusive line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {
                        "type": "integer",
                        "description": "First line to include (1-based)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to include (inclusive)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file on disk. Creates missing parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Text to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace the first occurrence of old_text with new_text in a file. old_text must match exactly and appear exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace (must appear exactly once)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace old_text with",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_path",
            "description": "Soft-delete a file or directory by moving it to a trash folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_path",
            "description": "Move or rename a file or directory. Creates missing parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "Current path"},
                    "dest": {"type": "string", "description": "New path"},
                },
                "required": ["src", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: current directory)",
                    }
                },
            },
        },
    },
]
