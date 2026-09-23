"""The incremental result writer: holds the document, flushes valid JSON as it grows."""

import json
import os
import time


class ResultWriter:
    """Holds the result document and writes valid JSON to disk after every update.

    With a path, each event/console line rewrites the file so partial results are
    always readable. Without a path, nothing is written; read document() instead.
    """

    def __init__(self, path=None, prompt=""):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.prompt = prompt
        self.events = []
        self.tool_calls = []
        self.console_lines = []
        self.total_tokens = 0
        self.output = ""
        self.prompt_length = 0
        self._start = time.time()

    def event(self, text):
        self.events.append(text)
        self._write()

    def add_assistant(self, message):
        """Record one assistant message as readable events (the user prompt is not an event)."""
        if message.get("reasoning_content"):
            self.event(f"reasoning> {message['reasoning_content']}")
        for tc in message.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.dumps(
                    json.loads(fn.get("arguments") or "{}"), ensure_ascii=False
                )
            except json.JSONDecodeError:
                args = fn.get("arguments") or "{}"  # handler reports these to the model
            self.event(f"tool> {fn['name']} {args}")
        if message.get("content"):
            self.event(f"content> {message['content']}")

    def tool_call(self, record):
        self.tool_calls.append(record)

    def console_line(self, line):
        self.console_lines.append(line)
        self._write()

    def add_tokens(self, n):
        self.total_tokens += n

    def document(self):
        """The current result document as a dict (same schema as the JSON file)."""
        return {
            "console": "\n".join(self.console_lines),
            "prompt": self.prompt,
            "output": self.output,
            "events": list(self.events),
            "tool_calls": list(self.tool_calls),
            "prompt_length": self.prompt_length,
            "total_tokens": self.total_tokens,
            "execution_time": round(time.time() - self._start, 1),
        }

    def _write(self):
        if self.path:
            with open(self.path, "w") as f:
                f.write(json.dumps(self.document(), indent=2) + "\n")

    def finish(self, output, prompt_length):
        """Record the final output and prompt length, then flush."""
        self.output = output
        self.prompt_length = prompt_length
        self._write()
