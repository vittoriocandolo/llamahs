# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The bash tool: the model's arguments read as meant, a command killed where it runs, its output cut to fit."""

from __future__ import annotations

import subprocess

import pytest

import tool
from settings import CONTAINER, TOOL_KILL_GRACE_SEC, TOOL_RESULT_CHARS, TOOL_TIMEOUT_MAX, TOOL_TIMEOUT_SEC


def test_arguments_are_read_as_the_model_means_them() -> None:
    """Read a JSON object, a JSON string or a bare command, and nothing from anything else."""
    assert tool.parse_args({"command": "ls", 1: 2}) == {"command": "ls", "1": 2}
    assert tool.parse_args('{"command": "pwd"}') == {"command": "pwd"}
    assert tool.parse_args("echo hi") == {"command": "echo hi"}
    assert tool.parse_args('"just a string"') == {"command": '"just a string"'}
    assert tool.parse_args(None) == {}


DEFAULT = 60


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        (30, 30),
        ("45", 45),
        (2.9, 2),
        ("soon", DEFAULT),
        (None, DEFAULT),
        ([], DEFAULT),
        (True, DEFAULT),
        (float("inf"), DEFAULT),
    ],
)
def test_a_timeout_is_a_whole_number_or_the_default(value: object, seconds: int) -> None:
    """Take an integer, a float or a numeric string as a number, and fall back on anything else."""
    assert tool.as_int(value, DEFAULT) == seconds


def test_output_is_text_whatever_it_came_as() -> None:
    """Decode bytes as UTF-8, replacing what is not, and read nothing as an empty text."""
    assert tool.as_text(None) == ""
    assert tool.as_text("ok") == "ok"
    assert tool.as_text("crème".encode() + bytes([0xFF])) == "crème\N{REPLACEMENT CHARACTER}"
    assert tool.as_text(memoryview(b"x")) == "x"
    assert tool.as_text(7) == "7"


def test_long_output_keeps_its_start_and_end_and_says_what_was_left_out() -> None:
    """Cut a long text to its head and tail, with the count of characters omitted between them."""
    assert tool.compact_text("  short  ", 100) == "short"
    cut = tool.compact_text("a" * 50 + "b" * 50, 20)
    assert cut == "a" * 10 + "\n... [80 chars omitted] ...\n" + "b" * 10
    assert tool.format_result_section("stdout", "   ", 10) == []
    assert tool.format_result_section("stdout", "x" * 30, 10) == [
        "[stdout_chars]=30",
        "[stdout]",
        tool.compact_text("x" * 30, 10),
    ]
    assert tool.format_result(0, "out", "", 10) == "[exit_code]=0\n[stdout]\nout"


Run = list[tuple[tuple[str, ...], dict[str, object]]]


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> Run:
    """Record the Docker commands the tool runs, answering each with success."""
    calls: Run = []

    def docker(*args: str, **options: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, options))
        return subprocess.CompletedProcess(args, 0, "done\n", "")

    monkeypatch.setattr(tool, "docker", docker)
    monkeypatch.setattr("runtime.container_user", lambda: "1000:1000")
    return calls


def test_a_command_runs_under_timeout_in_the_container(ran: Run) -> None:
    """Run the command in the container as the caller, under `timeout`, with a later local backstop."""
    assert tool.bash_exec("sleep 1", 30) == "[exit_code]=0\n[stdout]\ndone"
    ((args, options),) = ran
    assert args == (
        "exec",
        "-i",
        "-u",
        "1000:1000",
        "-w",
        "/workspace",
        CONTAINER,
        "timeout",
        f"--kill-after={TOOL_KILL_GRACE_SEC}",
        "30",
        "bash",
        "-lc",
        "sleep 1",
    )
    assert options == {"check": False, "capture_output": True, "timeout": 30 + 2 * TOOL_KILL_GRACE_SEC}


def test_a_timeout_is_held_between_one_second_and_the_most_allowed(ran: Run) -> None:
    """Raise a timeout below a second to one, and lower one above the most allowed to it."""
    tool.bash_exec("x", 0)
    tool.bash_exec("x", 10**9)
    assert [args[9] for args, _ in ran] == ["1", str(TOOL_TIMEOUT_MAX)]


def test_a_command_stopped_in_the_container_says_it_timed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report exit status 124 from `timeout` as a command that outlived its time."""
    monkeypatch.setattr(
        tool,
        "docker",
        lambda *args, **_: subprocess.CompletedProcess(args, tool.TIMED_OUT, "partial", None),
    )
    assert tool.bash_exec("sleep 99", 5) == "[exit_code]=124\n[stdout]\npartial\n[stderr]\nTimed out after 5s."


def test_a_container_that_does_not_answer_is_given_up_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give up at the local backstop, keeping whatever output came, and say why."""

    def hangs(*args: str, **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(list(args), 15, output=b"some", stderr=None)

    monkeypatch.setattr(tool, "docker", hangs)
    backstop = 5 + 2 * TOOL_KILL_GRACE_SEC
    assert tool.bash_exec("x", 5) == f"[exit_code]=124\n[stdout]\nsome\n[stderr]\nTimed out after {backstop}s."


def test_docker_that_cannot_run_is_an_answer_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the model with the failure when Docker itself cannot be run."""

    def missing(*_args: str, **_: object) -> subprocess.CompletedProcess[str]:
        msg = "docker: not found"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(tool, "docker", missing)
    assert tool.bash_exec("x", 5) == "[exit_code]=1\n[stderr]\nbash_exec failed: docker: not found"


def test_a_tool_call_runs_only_the_bash_tool(ran: Run) -> None:
    """Run a bash_exec call with its own or the default timeout, and refuse any other tool."""
    unknown = "[exit_code]=1\n[stderr]\nUnknown tool."
    assert tool.run_tool_call({"function": {"name": "rm_rf"}}) == unknown
    assert tool.run_tool_call({"function": "bash_exec"}) == unknown
    tool.run_tool_call({"function": {"name": "bash_exec", "arguments": '{"command": "ls", "timeout_sec": 7}'}})
    tool.run_tool_call({"function": {"name": "bash_exec", "arguments": "pwd"}})
    assert [(args[-1], args[9]) for args, _ in ran] == [("ls", "7"), ("pwd", str(TOOL_TIMEOUT_SEC))]
    assert len(tool.format_result(0, "x" * (TOOL_RESULT_CHARS * 2), "", TOOL_RESULT_CHARS)) < TOOL_RESULT_CHARS * 2
