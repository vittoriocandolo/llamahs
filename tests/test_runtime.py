# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The runtime: profiles, the server's address and lifecycle, the image and the locked-down container."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Self

import pytest

import runtime
from settings import CONTAINER, DEFAULT_PORT, IMAGE, IMAGE_SOURCE_HASH_LABEL

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .conftest import Server

Run = list[tuple[tuple[str, ...], dict[str, object]]]


def test_a_profile_is_a_command_line_and_nothing_less(tmp_path: Path) -> None:
    """Read a profile's command, and refuse a profile that is missing or empty."""
    (tmp_path / "profiles").mkdir()
    with pytest.raises(FileNotFoundError, match="missing profile"):
        runtime.load_command(tmp_path, "none")
    (tmp_path / "profiles" / "empty").write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty profile"):
        runtime.load_command(tmp_path, "empty")
    (tmp_path / "profiles" / "fast").write_text("llama-server -m m.gguf\n", encoding="utf-8")
    assert runtime.load_command(tmp_path, "fast") == "llama-server -m m.gguf"


@pytest.mark.parametrize(
    ("command", "port"),
    [
        ("llama-server -m m.gguf --port 9000", "9000"),
        ("llama-server --port=9001 -m m.gguf", "9001"),
        ("llama-server -m m.gguf", str(DEFAULT_PORT)),
        ("llama-server --ports 9002", str(DEFAULT_PORT)),
    ],
)
def test_the_server_is_reached_on_this_machine_at_its_port(command: str, port: str) -> None:
    """Find the command's --port, in either form, or the default one."""
    assert runtime.base_url(command) == f"http://127.0.0.1:{port}/v1"


def test_the_container_runs_as_the_caller_except_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as the caller's user and group where there are such, and leave it to Docker on Windows."""
    if sys.platform != "win32":
        assert runtime.container_user() == f"{os.getuid()}:{os.getgid()}"
        assert runtime.exec_args() == ["exec", "-i", "-u", runtime.container_user()]
    monkeypatch.setattr(os, "name", "nt")
    assert runtime.container_user() is None
    assert runtime.exec_args() == ["exec", "-i"]


def test_each_server_run_logs_to_its_own_file(tmp_path: Path) -> None:
    """Name the log by the local time to the microsecond, in the log folder."""
    path = runtime.server_log_path(tmp_path)
    assert path.parent == tmp_path / "logs"
    assert path.name.startswith("llama-server-")
    assert path.suffix == ".log"


def test_docker_is_found_on_path_or_said_to_be_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run Docker from PATH, checked by default, and say clearly when it is not installed."""
    seen: list[tuple[list[str], dict[str, object]]] = []

    def run(args: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        seen.append((args, options))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(subprocess, "run", run)
    runtime.docker("ps")
    runtime.docker("ps", check=False, capture_output=True, input="x", timeout=3)
    assert seen[0] == (
        ["/usr/bin/docker", "ps"],
        {
            "check": True,
            "text": True,
            "capture_output": False,
            "input": None,
            "timeout": None,
            "stdout": None,
            "stderr": None,
        },
    )
    assert seen[1][1] | {} == {
        "check": False,
        "text": True,
        "capture_output": True,
        "input": "x",
        "timeout": 3,
        "stdout": None,
        "stderr": None,
    }
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="needs Docker"):
        runtime.docker("ps")


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> Run:
    """Record the Docker commands the runtime runs, answering each with success and no output."""
    calls: Run = []

    def docker(*args: str, **options: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, options))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(runtime, "docker", docker)
    monkeypatch.setattr(runtime, "container_user", lambda: "1000:1000")
    return calls


def test_files_in_the_container_are_read_and_written_through_python(
    ran: Run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read a file's text, or nothing when it cannot be read, and write one from standard input."""
    runtime.write_container_text("/workspace/notes.md", "hello")
    args, options = ran[0]
    assert args[:6] == ("exec", "-i", "-u", "1000:1000", CONTAINER, "python3")
    assert args[-1] == "/workspace/notes.md"
    assert options == {"input": "hello"}
    monkeypatch.setattr(
        runtime,
        "docker",
        lambda *args, **_: subprocess.CompletedProcess(args, 0, "text", ""),
    )
    assert runtime.read_container_text("/workspace/notes.md") == "text"
    monkeypatch.setattr(
        runtime,
        "docker",
        lambda *args, **_: subprocess.CompletedProcess(args, 1, "", "no container"),
    )
    assert runtime.read_container_text("/workspace/notes.md") == ""


def test_the_image_is_built_only_when_its_files_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hash the Dockerfile and .dockerignore, and build only when the image's label differs."""
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    first = runtime.image_source_hash(tmp_path)
    (tmp_path / ".dockerignore").write_text("logs/\n", encoding="utf-8")
    assert runtime.image_source_hash(tmp_path) != first
    built: list[tuple[str, ...]] = []
    labels: dict[str, str | None] = {"value": runtime.image_source_hash(tmp_path)}
    monkeypatch.setattr(runtime, "image_label", lambda _image, _name: labels["value"])
    monkeypatch.setattr(runtime, "docker", lambda *args, **_: built.append(args))
    runtime.ensure_image(tmp_path)
    assert built == []
    labels["value"] = None
    runtime.ensure_image(tmp_path)
    ((args),) = built
    assert args[0] == "build"
    assert f"{IMAGE_SOURCE_HASH_LABEL}={runtime.image_source_hash(tmp_path)}" in args
    assert IMAGE in args


@pytest.mark.parametrize(
    ("status", "stdout", "label"),
    [
        (1, "", None),
        (0, "not json", None),
        (0, "[]", None),
        (0, "[1]", None),
        (0, json.dumps([{"Config": None}]), None),
        (0, json.dumps([{"Config": {"Labels": None}}]), None),
        (0, json.dumps([{"Config": {"Labels": {IMAGE_SOURCE_HASH_LABEL: 7}}}]), None),
        (0, json.dumps([{"Config": {"Labels": {IMAGE_SOURCE_HASH_LABEL: "abc"}}}]), "abc"),
    ],
)
def test_an_image_label_is_read_whatever_docker_answers(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    stdout: str,
    label: str | None,
) -> None:
    """Find the label in Docker's description of the image, or None for anything that is not it."""
    monkeypatch.setattr(
        runtime,
        "docker",
        lambda *args, **_: subprocess.CompletedProcess(args, status, stdout, ""),
    )
    assert runtime.image_label(IMAGE, IMAGE_SOURCE_HASH_LABEL) == label


def test_the_container_is_locked_down(tmp_path: Path, ran: Run) -> None:
    """Start the container afresh with no network, a read-only root, no capabilities and only /workspace kept."""
    runtime.start_container(tmp_path)
    (removed, _), (started, _) = ran
    assert removed == ("rm", "-f", CONTAINER)
    for flags in (
        ("--read-only",),
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--user", "1000:1000"),
        ("-v", f"{(tmp_path / 'workspace').resolve()}:/workspace"),
    ):
        at = started.index(flags[0])
        assert started[at : at + len(flags)] == flags
    assert (tmp_path / "workspace").is_dir()


def test_the_container_runs_without_a_user_on_windows(
    tmp_path: Path,
    ran: Run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the user to Docker where there are no user ids."""
    monkeypatch.setattr(runtime, "container_user", lambda: None)
    runtime.start_container(tmp_path)
    assert "--user" not in ran[-1][0]


def test_the_server_runs_in_a_login_shell_or_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the profile's command through bash, or as a command line on Windows, its output to the log."""
    started: list[tuple[object, dict[str, object]]] = []

    def popen(args: object, **options: object) -> object:
        started.append((args, options))
        return object()  # a stand-in: only the call is checked

    monkeypatch.setattr(subprocess, "Popen", popen)
    log = tmp_path / "logs" / "server.log"
    runtime.start_server(log, "llama-server -m m.gguf")
    assert started[0][0] == ["/bin/bash", "-lc", "llama-server -m m.gguf"]
    assert log.is_file()
    monkeypatch.setattr(os, "name", "nt")
    runtime.start_server(log, "llama-server -m m.gguf")
    assert started[1][0] == "llama-server -m m.gguf"
    assert started[1][1]["shell"] is True


class Proc:
    """A server process that has exited or not, and stops when told, or only when killed."""

    def __init__(self, *, exited: bool = False, stubborn: bool = False) -> None:
        """Start running, or already exited."""
        self.exited, self.stubborn = exited, stubborn
        self.signals: list[str] = []

    def poll(self) -> int | None:
        """Return 0 once it has exited, else None."""
        return 0 if self.exited else None

    def terminate(self) -> None:
        """Ask it to stop."""
        self.signals.append("terminate")

    def kill(self) -> None:
        """Kill it."""
        self.signals.append("kill")
        self.exited = True

    def wait(self, timeout: float) -> int:
        """Wait for it, which a stubborn process makes time out until it is killed."""
        if self.stubborn and not self.exited:
            raise subprocess.TimeoutExpired(cmd=["llama-server"], timeout=timeout)
        self.exited = True
        return 0


def test_the_server_is_stopped_and_killed_if_it_will_not_stop() -> None:
    """Leave an exited server alone, stop a running one, and kill one that will not stop."""
    done = Proc(exited=True)
    runtime.stop_server(done)  # type: ignore[arg-type]
    assert done.signals == []
    polite = Proc()
    runtime.stop_server(polite)  # type: ignore[arg-type]
    assert polite.signals == ["terminate"]
    stubborn = Proc(stubborn=True)
    runtime.stop_server(stubborn)  # type: ignore[arg-type]
    assert stubborn.signals == ["terminate", "kill"]


def test_the_server_is_waited_for_until_it_answers(
    server: Server,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait while the server refuses, ask again after a pause, and return once it answers."""
    answers: Iterator[tuple[int, object]] = iter([(503, {}), (200, {"data": []})])
    server.answers(lambda _path, _body: next(answers))
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    runtime.wait_ready(Proc(), server.url, tmp_path / "log")  # type: ignore[arg-type]
    assert len(pauses) == 1
    assert [path for _, path, _ in server.seen] == ["/v1/models", "/v1/models"]


def test_a_server_that_exits_or_never_answers_is_given_up_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail at once, naming the log, if the server exits; fail at the deadline if it never answers."""
    log = tmp_path / "server.log"
    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        runtime.wait_ready(Proc(exited=True), "http://127.0.0.1:9/v1", log)  # type: ignore[arg-type]
    clock = iter([0.0, 0.0, 10**6])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match=str(log).replace("\\", "\\\\")):
        runtime.wait_ready(Proc(), "http://127.0.0.1:9/v1", log)  # type: ignore[arg-type]


def test_an_answer_outside_success_is_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep waiting on an answer that is neither success nor an error, as a redirect left unfollowed would be."""

    class Redirect:
        status = 302

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    statuses = iter([Redirect(), None])

    def urlopen(_url: str, timeout: float) -> object:
        assert timeout > 0
        reply = next(statuses)
        if reply is None:
            raise urllib.error.URLError(reason="refused")
        return reply

    clock = iter([0.0, 0.0, 0.0, 10**6])
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    with pytest.raises(TimeoutError):
        runtime.wait_ready(Proc(), "http://127.0.0.1:9/v1", tmp_path / "log")  # type: ignore[arg-type]
    assert len(pauses) == len(["redirect", "refused"])
