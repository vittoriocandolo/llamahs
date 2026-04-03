from __future__ import annotations

import json
import subprocess
from typing import Any

from runtime import container_user, docker
from settings import CONTAINER, TOOL_RESULT_CHARS, TOOL_TIMEOUT_MAX, TOOL_TIMEOUT_SEC

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Run one bash command in an Ubuntu container at /workspace. Tools: bash, python3, rg, git, jq. No network.",
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
    }
]


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def parse_args(raw: Any) -> dict[str, Any]:
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
    text = text.strip()
    if len(text) <= sample_chars:
        return text
    head = max(1, sample_chars // 2)
    tail = max(1, sample_chars - head)
    omitted = len(text) - head - tail
    return "\n".join(
        [
            text[:head].rstrip(),
            f"... [{omitted} chars omitted] ...",
            text[-tail:].lstrip(),
        ]
    )


def format_result_section(name: str, value: str, cap: int) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if len(text) <= cap:
        return [f"[{name}]", text]
    return [f"[{name}_chars]={len(text)}", f"[{name}]", compact_text(text, cap)]


def format_result(exit_code: int, stdout: str, stderr: str, cap: int) -> str:
    lines = [f"[exit_code]={exit_code}"]
    lines.extend(format_result_section("stdout", stdout, cap))
    lines.extend(format_result_section("stderr", stderr, cap))
    return "\n".join(lines)


def bash_exec(command: str, timeout_sec: int) -> str:
    timeout = max(1, min(timeout_sec, TOOL_TIMEOUT_MAX))
    args = ["exec", "-i"]
    user = container_user()
    if user is not None:
        args.extend(["-u", user])
    args.extend(["-w", "/workspace", CONTAINER, "bash", "-lc", command])
    try:
        result = docker(*args, check=False, capture_output=True, timeout=timeout)
        return format_result(
            result.returncode,
            result.stdout or "",
            result.stderr or "",
            TOOL_RESULT_CHARS,
        )
    except subprocess.TimeoutExpired as exc:
        return format_result(
            124,
            as_text(exc.stdout),
            as_text(exc.stderr) + f"\nTimed out after {timeout}s.",
            TOOL_RESULT_CHARS,
        )
    except Exception as exc:
        return format_result(1, "", f"bash_exec failed: {exc}", TOOL_RESULT_CHARS)


def run_tool_call(call: dict[str, Any]) -> str:
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "bash_exec":
        return "[exit_code]=1\n[stderr]\nUnknown tool."
    args = parse_args(function.get("arguments"))
    return bash_exec(
        str(args.get("command", "")),
        as_int(args.get("timeout_sec", TOOL_TIMEOUT_SEC), TOOL_TIMEOUT_SEC),
    )
