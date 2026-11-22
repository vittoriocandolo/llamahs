# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The one tool the model has: a bash command in the container, its output cut to what the model can read."""

from __future__ import annotations

import json
from time import monotonic

from runtime import Kept, capture, docker_path, exec_args
from settings import (
    CONTAINER,
    TOOL_KILL_GRACE_SEC,
    TOOL_RESULT_CHARS,
    TOOL_TIMEOUT_MAX,
    TOOL_TIMEOUT_SEC,
)

TIMED_OUT = 124  # the exit status of a command `timeout` stopped, as coreutils gives it

TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": (
                "Run one bash command in an Ubuntu container at /workspace. Tools: bash, python3, rg, git, jq. "
                "No network."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to run.",
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Seconds before the command is killed.",
                        "default": TOOL_TIMEOUT_SEC,
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def as_count(value: object, minimum: int = 0) -> int | None:
    """Return a value as a whole count at least `minimum`, or None if it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = int(value)
    except (ValueError, OverflowError):
        return None
    return number if number >= minimum else None


def parse_args(raw: object) -> dict[str, object]:
    """Return a tool call's arguments: its JSON object, or a string, bare or in JSON, taken as the command itself."""
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {"command": raw}
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        if isinstance(value, str):
            return {"command": value}
        return {"command": raw}
    return {}


NOTHING = Kept(b"", b"", 0)


def format_section(name: str, kept: Kept) -> list[str]:
    """Return one stream of a result, headed by its name, with its full length in bytes if it was cut."""
    omitted = kept.total - len(kept.head) - len(kept.tail)
    if omitted <= 0:
        text = (kept.head + kept.tail).decode("utf-8", errors="replace").strip()
        return [f"[{name}]", text] if text else []
    head = kept.head.decode("utf-8", errors="replace").rstrip()
    tail = kept.tail.decode("utf-8", errors="replace").lstrip()
    return [f"[{name}_bytes]={kept.total}", f"[{name}]", f"{head}\n... [{omitted} bytes omitted] ...\n{tail}"]


def format_result(exit_code: int, stdout: Kept, stderr: Kept, note: str = "") -> str:
    """Return a command's result as the model reads it: its exit status, its output, its errors, and a note."""
    lines = [f"[exit_code]={exit_code}", *format_section("stdout", stdout)]
    errors = format_section("stderr", stderr)
    if note:
        errors = [*(errors or ["[stderr]"]), note]
    return "\n".join([*lines, *errors])


def bash_exec(command: str, timeout_sec: int) -> str:
    """Run one command in the container, killed there once it outlives its timeout.

    The command runs under coreutils `timeout` inside the container, so a command that outlives its time is
    stopped where it runs (exit status 124), not only the local `docker exec` client. The local timeout, a little
    later, is a backstop for a container that no longer answers. Of each output stream only the start and the end
    are kept, as the model reads them, so a command that prints without end costs the host nothing more.
    """
    timeout = max(1, min(timeout_sec, TOOL_TIMEOUT_MAX))
    args = [
        *exec_args(),
        "-w",
        "/workspace",
        CONTAINER,
        "timeout",
        f"--kill-after={TOOL_KILL_GRACE_SEC}",
        str(timeout),
        "bash",
        "-lc",
        command,
    ]
    backstop = timeout + 2 * TOOL_KILL_GRACE_SEC
    head = TOOL_RESULT_CHARS // 2
    started = monotonic()
    try:
        run = capture([docker_path(), *args], backstop, head, TOOL_RESULT_CHARS - head)
    except OSError as exc:  # Docker itself could not be run
        return format_result(1, NOTHING, NOTHING, f"bash_exec failed: {exc}")
    if run.timed_out:
        return format_result(TIMED_OUT, run.stdout, run.stderr, f"Timed out after {backstop}s.")
    # 124 is also an ordinary exit status: it means `timeout` stopped the command only once its time has passed
    stopped = run.code == TIMED_OUT and monotonic() - started >= timeout
    return format_result(run.code, run.stdout, run.stderr, f"Timed out after {timeout}s." if stopped else "")


def run_tool_call(call: dict[str, object]) -> str:
    """Run a tool call the model made, or say that the tool is unknown."""
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "bash_exec":
        return "[exit_code]=1\n[stderr]\nUnknown tool."
    args = parse_args(function.get("arguments"))
    return bash_exec(
        str(args.get("command", "")),
        as_count(args.get("timeout_sec")) or TOOL_TIMEOUT_SEC,
    )
