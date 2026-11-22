# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The bash tool: the model's arguments read as meant, a command killed where it runs, its output cut to fit."""

from __future__ import annotations

import pytest

import tool
from runtime import Captured, Kept
from settings import CONTAINER, TOOL_KILL_GRACE_SEC, TOOL_RESULT_CHARS, TOOL_TIMEOUT_MAX, TOOL_TIMEOUT_SEC


def test_arguments_are_read_as_the_model_means_them() -> None:
    """Read a JSON object, a JSON string or a bare command, and nothing from anything else."""
    assert tool.parse_args({"command": "ls", 1: 2}) == {"command": "ls", "1": 2}
    assert tool.parse_args('{"command": "pwd"}') == {"command": "pwd"}
    assert tool.parse_args("echo hi") == {"command": "echo hi"}
    assert tool.parse_args('"just a string"') == {"command": "just a string"}
    assert tool.parse_args("[1, 2]") == {"command": "[1, 2]"}
    assert tool.parse_args(None) == {}


def test_output_keeps_its_start_and_end_and_says_how_much_was_left_out() -> None:
    """Show a stream whole when nothing was dropped, and its start and end with the bytes left out otherwise."""
    assert tool.format_section("stdout", Kept(b"  ", b"", 2)) == []
    assert tool.format_section("stdout", Kept(b"cr\xc3", b"\xa8me ", 7)) == ["[stdout]", "crème"]
    assert tool.format_section("stdout", Kept(b"aaa ", b" bbb", 100)) == [
        "[stdout_bytes]=100",
        "[stdout]",
        "aaa\n... [92 bytes omitted] ...\nbbb",
    ]
    out = Kept(b"out", b"", 3)
    assert tool.format_result(0, out, tool.NOTHING) == "[exit_code]=0\n[stdout]\nout"
    assert tool.format_result(1, tool.NOTHING, tool.NOTHING, "why") == "[exit_code]=1\n[stderr]\nwhy"
    assert tool.format_result(1, tool.NOTHING, Kept(b"err", b"", 3), "why") == "[exit_code]=1\n[stderr]\nerr\nwhy"


Run = list[tuple[list[str], float, int, int]]


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> Run:
    """Record the programs the tool runs, answering each with success."""
    calls: Run = []

    def capture(args: list[str], timeout: float, head: int, tail: int) -> Captured:
        calls.append((args, timeout, head, tail))
        return Captured(0, Kept(b"done\n", b"", 5), tool.NOTHING, timed_out=False)

    monkeypatch.setattr(tool, "capture", capture)
    monkeypatch.setattr(tool, "docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr("runtime.container_user", lambda: "1000:1000")
    return calls


def test_a_command_runs_under_timeout_in_the_container(ran: Run) -> None:
    """Run the command in the container as the caller, under `timeout`, with a later local backstop."""
    assert tool.bash_exec("sleep 1", 30) == "[exit_code]=0\n[stdout]\ndone"
    ((args, timeout, head, tail),) = ran
    assert args == [
        "/usr/bin/docker",
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
    ]
    assert timeout == 30 + 2 * TOOL_KILL_GRACE_SEC
    assert head + tail == TOOL_RESULT_CHARS


def test_a_timeout_is_held_between_one_second_and_the_most_allowed(ran: Run) -> None:
    """Raise a timeout below a second to one, and lower one above the most allowed to it."""
    tool.bash_exec("x", 0)
    tool.bash_exec("x", 10**9)
    assert [args[10] for args, *_ in ran] == ["1", str(TOOL_TIMEOUT_MAX)]


def finished(code: int, monkeypatch: pytest.MonkeyPatch, elapsed: float) -> None:
    """Make every run end with `code` after `elapsed` seconds."""
    clock = iter([0.0, elapsed])
    monkeypatch.setattr("tool.monotonic", lambda: next(clock))
    monkeypatch.setattr(tool, "docker_path", lambda: "docker")
    monkeypatch.setattr(
        tool,
        "capture",
        lambda *_: Captured(code, Kept(b"partial", b"", 7), tool.NOTHING, timed_out=False),
    )


def test_a_command_stopped_in_the_container_says_it_timed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report exit status 124 from `timeout`, once the command's time has passed, as a command that outlived it."""
    finished(tool.TIMED_OUT, monkeypatch, 5.0)
    assert tool.bash_exec("sleep 99", 5) == "[exit_code]=124\n[stdout]\npartial\n[stderr]\nTimed out after 5s."


def test_a_command_that_exits_124_itself_did_not_time_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report exit status 124 before the command's time as the command's own status."""
    finished(tool.TIMED_OUT, monkeypatch, 0.1)
    assert tool.bash_exec("exit 124", 5) == "[exit_code]=124\n[stdout]\npartial"


def test_a_container_that_does_not_answer_is_given_up_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give up at the local backstop, keeping whatever output came, and say why."""
    monkeypatch.setattr(tool, "docker_path", lambda: "docker")
    monkeypatch.setattr(tool, "capture", lambda *_: Captured(-9, Kept(b"some", b"", 4), tool.NOTHING, timed_out=True))
    backstop = 5 + 2 * TOOL_KILL_GRACE_SEC
    assert tool.bash_exec("x", 5) == f"[exit_code]=124\n[stdout]\nsome\n[stderr]\nTimed out after {backstop}s."


def test_docker_that_cannot_run_is_an_answer_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the model with the failure when Docker itself cannot be found."""

    def missing() -> str:
        msg = "docker: not found"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(tool, "docker_path", missing)
    assert tool.bash_exec("x", 5) == "[exit_code]=1\n[stderr]\nbash_exec failed: docker: not found"


def test_a_tool_call_runs_only_the_bash_tool(ran: Run) -> None:
    """Run a bash_exec call with its own or the default timeout, and refuse any other tool."""
    unknown = "[exit_code]=1\n[stderr]\nUnknown tool."
    assert tool.run_tool_call({"function": {"name": "rm_rf"}}) == unknown
    assert tool.run_tool_call({"function": "bash_exec"}) == unknown
    tool.run_tool_call({"function": {"name": "bash_exec", "arguments": '{"command": "ls", "timeout_sec": 7}'}})
    tool.run_tool_call({"function": {"name": "bash_exec", "arguments": "pwd"}})
    tool.run_tool_call({"function": {"name": "bash_exec", "arguments": '{"command": "id", "timeout_sec": "soon"}'}})
    assert [(args[-1], args[10]) for args, *_ in ran] == [
        ("ls", "7"),
        ("pwd", str(TOOL_TIMEOUT_SEC)),
        ("id", str(TOOL_TIMEOUT_SEC)),
    ]
