# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The one tool the model has: a bash command in the container, its output cut to what the model can read."""

from __future__ import annotations

import json
import subprocess

from runtime import docker, exec_args
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


def as_int(value: object, default: int) -> int:
    """Return a value as an integer, or the default if it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def as_text(value: object) -> str:
    """Return a command's output as text, whether it came as text, bytes or nothing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def parse_args(raw: object) -> dict[str, object]:
    """Return a tool call's arguments: its JSON object, or a bare string taken as the command itself."""
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {"command": raw}
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        return {"command": raw}
    return {}


def compact_text(text: str, sample_chars: int) -> str:
    """Return a text cut to about `sample_chars`: its start and its end, and how much was left out between."""
    text = text.strip()
    if len(text) <= sample_chars:
        return text
    head = max(1, sample_chars // 2)
    tail = max(1, sample_chars - head)
    omitted = len(text) - head - tail
    return "\n".join([text[:head].rstrip(), f"... [{omitted} chars omitted] ...", text[-tail:].lstrip()])


def format_result_section(name: str, value: str, cap: int) -> list[str]:
    """Return one stream of a result, headed by its name, with its full length if it was cut."""
    text = value.strip()
    if not text:
        return []
    if len(text) <= cap:
        return [f"[{name}]", text]
    return [f"[{name}_chars]={len(text)}", f"[{name}]", compact_text(text, cap)]


def format_result(exit_code: int, stdout: str, stderr: str, cap: int) -> str:
    """Return a command's result as the model reads it: its exit status, then its output and its errors."""
    lines = [f"[exit_code]={exit_code}"]
    lines.extend(format_result_section("stdout", stdout, cap))
    lines.extend(format_result_section("stderr", stderr, cap))
    return "\n".join(lines)


def bash_exec(command: str, timeout_sec: int) -> str:
    """Run one command in the container, killed there once it outlives its timeout.

    The command runs under coreutils `timeout` inside the container, so a command that outlives its time is
    stopped where it runs (exit status 124), not only the local `docker exec` client. The local timeout, a little
    later, is a backstop for a container that no longer answers.
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
    try:
        result = docker(*args, check=False, capture_output=True, timeout=backstop)
    except subprocess.TimeoutExpired as exc:
        stderr = as_text(exc.stderr) + f"\nTimed out after {backstop}s."
        return format_result(TIMED_OUT, as_text(exc.stdout), stderr, TOOL_RESULT_CHARS)
    except OSError as exc:  # Docker itself could not be run
        return format_result(1, "", f"bash_exec failed: {exc}", TOOL_RESULT_CHARS)
    stderr = result.stderr or ""
    if result.returncode == TIMED_OUT:
        stderr += f"\nTimed out after {timeout}s."
    return format_result(result.returncode, result.stdout or "", stderr, TOOL_RESULT_CHARS)


def run_tool_call(call: dict[str, object]) -> str:
    """Run a tool call the model made, or say that the tool is unknown."""
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "bash_exec":
        return "[exit_code]=1\n[stderr]\nUnknown tool."
    args = parse_args(function.get("arguments"))
    return bash_exec(
        str(args.get("command", "")),
        as_int(args.get("timeout_sec", TOOL_TIMEOUT_SEC), TOOL_TIMEOUT_SEC),
    )
