# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The conversation: requests, compaction, tool rounds, failed turns dropped, and the program's exit statuses."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import main
from settings import DEFAULT_MODEL_ID, MAX_TOOL_ROUNDS, OUTPUT_MAX_TOKENS, SESSION_CONTEXT_PATH, SUMMARY_MAX_TOKENS

if TYPE_CHECKING:
    from .conftest import Server

ROOT = Path(__file__).resolve().parent.parent
USAGE, FAILED, INTERRUPTED = 2, 1, 130  # main's exit statuses


def answer(content: object = None, calls: object = None, timings: object = None) -> dict[str, object]:
    """Return a chat answer as llama-server sends it."""
    message: dict[str, object] = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    out: dict[str, object] = {"choices": [{"message": message}]}
    if timings is not None:
        out["timings"] = timings
    return out


def call(command: str, ident: str = "c1") -> dict[str, object]:
    """Return a bash_exec call as the model makes it."""
    return {
        "id": ident,
        "type": "function",
        "function": {"name": "bash_exec", "arguments": f'{{"command": "{command}"}}'},
    }


def test_requests_send_json_and_read_json(server: Server) -> None:
    """GET without a body, POST JSON, and turn an error status into its body's message."""
    server.answers(lambda path, body: (200, {"path": path, "body": body}))
    assert main.request_json(f"{server.url}/models") == {"path": "/v1/models", "body": None}
    assert main.request_json(f"{server.url}/tokenize", {"content": "hi"}) == {
        "path": "/v1/tokenize",
        "body": {"content": "hi"},
    }
    assert [method for method, _, _ in server.seen] == ["GET", "POST"]
    server.answers(lambda _path, _body: (500, {"error": "out of memory"}))
    with pytest.raises(RuntimeError, match="out of memory"):
        main.request_json(f"{server.url}/x")
    server.answers(lambda _path, _body: (503, b""))
    with pytest.raises(RuntimeError, match="http 503"):
        main.request_json(f"{server.url}/x")


def test_an_answer_without_a_message_is_refused() -> None:
    """Return an answer's message, and refuse an answer of any other shape."""
    assert main.message_of(answer("hi"))["content"] == "hi"
    bad: object
    for bad in (None, {}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": "hi"}]}):
        with pytest.raises(TypeError, match="no message"):
            main.message_of(bad)
    assert main.text_of(None) == ""
    assert main.text_of("x") == "x"


@pytest.mark.parametrize(
    ("payload", "model"),
    [
        ({"data": [{"id": " qwen "}]}, "qwen"),
        ([{"id": "listed"}], "listed"),
        ({"data": [{"id": " "}], "id": "own"}, "own"),
        ({"data": [], "id": ""}, DEFAULT_MODEL_ID),
        ({"data": ["x"]}, DEFAULT_MODEL_ID),
        ("nonsense", DEFAULT_MODEL_ID),
    ],
)
def test_the_model_is_named_by_the_server_or_by_default(server: Server, payload: object, model: str) -> None:
    """Take the first model the server lists, its own id, or the default."""
    server.answers(lambda _path, _body: (200, payload))
    assert main.model_id(server.url) == model


def test_the_context_size_comes_from_the_properties_or_the_slots(server: Server) -> None:
    """Read n_ctx from the properties, else from the first slot that has one, else say nothing."""
    replies: dict[str, tuple[int, object]] = {}
    server.answers(lambda path, _body: replies.get(path, (404, {})))
    props, slot = 4096, 2048
    replies["/v1/props"] = (200, {"default_generation_settings": {"n_ctx": props}})
    assert main.context_limit(server.url) == props
    replies["/v1/props"] = (200, {"default_generation_settings": {"n_ctx": 0}})
    replies["/v1/slots"] = (200, ["x", {"n_ctx": "bad"}, {"n_ctx": slot}])
    assert main.context_limit(server.url) == slot
    replies["/v1/props"] = (500, {})
    replies["/v1/slots"] = (200, [{"n_ctx": None}])
    assert main.context_limit(server.url) is None
    replies["/v1/props"] = (200, {"default_generation_settings": None})
    replies["/v1/slots"] = (200, {"not": "a list"})
    assert main.context_limit(server.url) is None
    replies["/v1/slots"] = (500, {})
    assert main.context_limit(server.url) is None


@pytest.mark.parametrize(
    ("value", "count"),
    [("12", 12), (3.7, 3), (True, None), (None, None), ("x", None), (float("inf"), None), (-1, None)],
)
def test_counts_are_whole_numbers_or_nothing(value: object, count: int | None) -> None:
    """Read counts from numbers and numeric strings, and nothing from anything else or below the minimum."""
    assert main.as_count(value) == count
    assert main.as_count(0, 1) is None


def test_the_context_used_is_read_from_an_answers_timings() -> None:
    """Add the cached, prompt and predicted tokens, or say nothing if any is missing."""
    timings = {"cache_n": 10, "prompt_n": 5, "predicted_n": 2}
    assert main.context_usage(answer(timings=timings)) == sum(timings.values())
    assert main.context_usage(answer(timings={"cache_n": 10, "prompt_n": 5})) is None
    assert main.context_usage(answer()) is None
    assert main.context_usage("nonsense") is None


def test_the_prompt_is_counted_with_the_tools_description(server: Server) -> None:
    """Count the templated conversation and the tool's description, or nothing if any step fails."""
    replies: dict[str, tuple[int, object]] = {}
    server.answers(lambda path, _body: replies.get(path, (404, {})))
    tokens = [1, 2, 3]
    replies["/v1/apply-template"] = (200, {"prompt": "hello"})
    replies["/v1/tokenize"] = (200, {"tokens": tokens})
    assert main.prompt_tokens(server.url, []) == 2 * len(tokens)  # the conversation, then the tool's description
    replies["/v1/tokenize"] = (200, {"tokens": "x"})
    assert main.prompt_tokens(server.url, []) is None
    replies["/v1/apply-template"] = (200, {"prompt": 5})
    assert main.prompt_tokens(server.url, []) is None
    replies["/v1/apply-template"] = (500, {})
    assert main.prompt_tokens(server.url, []) is None
    replies["/v1/tokenize"] = (500, {})
    assert main.token_count(server.url, "x") is None


def test_only_the_tool_count_can_fail_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Say nothing when the conversation is counted but the tool's description is not."""
    counts = iter([4, None])
    monkeypatch.setattr(main, "request_json", lambda *_a, **_k: {"prompt": "p"})
    monkeypatch.setattr(main, "token_count", lambda *_a: next(counts))
    assert main.prompt_tokens("u", []) is None


class Container:
    """The container's files, as the harness reads and writes them, and whether writing fails."""

    def __init__(self) -> None:
        """Start empty."""
        self.files: dict[str, str] = {}
        self.broken = False

    def read(self, path: str) -> str:
        """Return a file, or nothing."""
        return self.files.get(path, "")

    def write(self, path: str, content: str) -> None:
        """Write a file, or fail as Docker would."""
        if self.broken:
            raise subprocess.CalledProcessError(1, "docker exec")
        self.files[path] = content


@pytest.fixture
def container(monkeypatch: pytest.MonkeyPatch) -> Container:
    """Stand in for the container's files."""
    box = Container()
    monkeypatch.setattr(main, "read_container_text", box.read)
    monkeypatch.setattr(main, "write_container_text", box.write)
    return box


def session(url: str, n_ctx: int | None = 100_000) -> main.Session:
    """Return a session with the repository's own prompts."""
    return main.Session(ROOT, url, "m", n_ctx)


def test_a_conversation_starts_from_its_compacted_context(container: Container) -> None:
    """Start with the context note when there is compacted context, else with nothing."""
    assert main.seed_messages() == []
    container.files[SESSION_CONTEXT_PATH] = "# Context\n"
    assert main.seed_messages() == [{"role": "assistant", "content": main.context_note()}]


def test_compaction_waits_for_the_contexts_end_unless_asked(
    server: Server,
    container: Container,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the conversation while it fits, and compact it once the next answer and summary would not."""
    server.answers(lambda _path, _body: (200, answer("# Context\n\nwork in progress")))
    talk: list[main.Message] = [{"role": "user", "content": "hi"}]
    assert main.maybe_compact_history(session(server.url), [], None) == []
    assert "nothing to compact" in capsys.readouterr().out
    room = OUTPUT_MAX_TOKENS + SUMMARY_MAX_TOKENS
    assert main.maybe_compact_history(session(server.url, room + 1000), talk, 999) is talk
    assert main.maybe_compact_history(session(server.url, None), talk, 1) is talk
    assert main.maybe_compact_history(session(server.url, room + 1000), talk, 1000) == [
        {"role": "assistant", "content": main.context_note()},
    ]
    assert container.files[SESSION_CONTEXT_PATH] == "# Context\n\nwork in progress\n"
    assert "context compacted" in capsys.readouterr().out


def test_compaction_counts_the_prompt_when_no_answer_said(
    server: Server,
    container: Container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask the server for the prompt's size when the last answer gave no usage."""
    monkeypatch.setattr(main, "prompt_tokens", lambda _url, _messages: None)
    talk: list[main.Message] = [{"role": "user", "content": "hi"}]
    assert main.maybe_compact_history(session(server.url), talk, None) is talk
    assert container.files == {}


def test_a_summary_merges_the_context_before_it(server: Server, container: Container) -> None:
    """Hand the earlier context to the model to merge, and keep it if the model says nothing."""
    container.files[SESSION_CONTEXT_PATH] = "# Context\n\nearlier"
    server.answers(lambda _path, _body: (200, answer("")))
    assert main.summarize_history(session(server.url), []) == "# Context\n\nearlier"
    (_, _, body) = server.seen[-1]
    assert isinstance(body, dict)
    assert "earlier" in str(body["messages"])
    del container.files[SESSION_CONTEXT_PATH]
    assert main.summarize_history(session(server.url), []) == "# Context\n\nNo durable context yet."


def test_a_compaction_that_fails_keeps_the_conversation(
    server: Server,
    container: Container,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the conversation as it was when the summary or its writing fails."""
    talk: list[main.Message] = [{"role": "user", "content": "hi"}]
    server.answers(lambda _path, _body: (500, {"error": "busy"}))
    assert main.maybe_compact_history(session(server.url), talk, None, force=True) is talk
    server.answers(lambda _path, _body: (200, answer("summary")))
    container.broken = True
    assert main.maybe_compact_history(session(server.url), talk, None, force=True) is talk
    assert capsys.readouterr().out.count("compaction skipped") == len(["summary failed", "write failed"])


@pytest.fixture
def typed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer the prompt with the lines a test puts here, then end the input."""
    lines: list[str] = []

    def read(_prompt: str) -> str:
        if not lines:
            raise EOFError
        return lines.pop(0)

    monkeypatch.setattr("builtins.input", read)
    return lines


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record the tool calls run, each answered with a fixed result."""
    ran: list[dict[str, object]] = []

    def run(call: dict[str, object]) -> str:
        ran.append(call)
        return "[exit_code]=0"

    monkeypatch.setattr(main, "run_tool_call", run)
    return ran


def last_chat(server: Server) -> dict[str, object]:
    """Return the body of the last chat request the server saw."""
    body = [b for _, path, b in server.seen if path == "/v1/chat/completions"][-1]
    assert isinstance(body, dict)
    return body


def replay(server: Server, replies: list[tuple[int, object]]) -> None:
    """Answer chat requests with these replies in turn, and the server's own questions as it would."""
    chats = iter(replies)

    def handle(path: str, _body: object) -> tuple[int, object]:
        if path == "/v1/chat/completions":
            return next(chats)
        return 404, {}

    server.answers(handle)


def test_a_turn_runs_the_tools_the_model_asks_for(
    server: Server,
    container: Container,
    typed: list[str],
    tools: list[dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the model's tool calls, hand back their results, and print its answer."""
    assert container.files == {}
    replay(server, [(200, answer(None, [call("ls")])), (200, answer("Done."))])
    typed.extend(["", "list the files", "exit"])
    main.repl(ROOT, server.url, "m")
    assert [c["id"] for c in tools] == ["c1"]
    out = capsys.readouterr().out
    assert "assistant> Done." in out
    messages = last_chat(server)["messages"]
    assert isinstance(messages, list)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]


def test_a_turn_stops_after_its_tool_rounds(
    server: Server,
    container: Container,
    typed: list[str],
    tools: list[dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop a turn that keeps calling tools after MAX_TOOL_ROUNDS rounds, and say so."""
    assert container.files == {}
    replay(server, [(200, answer("", [call("ls", f"c{i}")])) for i in range(MAX_TOOL_ROUNDS)])
    typed.extend(["loop", "quit"])
    main.repl(ROOT, server.url, "m")
    assert len(tools) == MAX_TOOL_ROUNDS
    assert "Stopped after max tool rounds." in capsys.readouterr().out


def test_a_failed_turn_is_dropped_and_the_session_goes_on(
    server: Server,
    container: Container,
    typed: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drop a turn whose request fails or whose answer is malformed, and answer the next one."""
    assert container.files == {}
    replay(
        server,
        [
            (500, {"error": "busy"}),
            (200, answer(None, [{"function": {"name": "bash_exec"}}])),
            (200, answer("")),
        ],
    )
    typed.extend(["first", "second", "third"])
    main.repl(ROOT, server.url, "m")
    out = capsys.readouterr().out
    assert out.count("request failed, nothing kept from this turn") == len(["error status", "malformed call"])
    assert "assistant> (no response)" in out
    assert last_chat(server)["messages"] == [{"role": "user", "content": "third"}]  # failed turns left nothing


def test_compact_on_request_and_leaving_on_ctrl_c(
    server: Server,
    container: Container,
    typed: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compact when asked, and leave quietly on Ctrl+C."""
    assert container.files == {}
    typed.append("/compact")

    def read(_prompt: str) -> str:
        if typed:
            return typed.pop(0)
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", read)
    main.repl(ROOT, server.url, "m")
    assert "nothing to compact" in capsys.readouterr().out


def test_tool_calls_keep_their_arguments_as_compact_json() -> None:
    """Write the calls' arguments as compact JSON, and refuse a call without an id."""
    message = main.tool_call_message("thinking", [call("ls")])
    assert message["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "bash_exec", "arguments": '{"command":"ls"}'}},
    ]
    with pytest.raises(TypeError, match="without an id"):
        main.function_of({"function": {"name": "bash_exec"}})


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for every step main takes, recording them; repl leaves at once."""
    steps: list[str] = []
    for name in ("ensure_image", "start_container", "remove_container"):
        monkeypatch.setattr(main, name, lambda *_a, name=name: steps.append(name))
    monkeypatch.setattr(main, "load_command", lambda _root, _name: "llama-server --port 9")
    monkeypatch.setattr(main, "server_log_path", lambda root: root / "logs" / "x.log")

    def start_server(_log: Path, _command: str) -> str:
        steps.append("start_server")
        return "proc"

    monkeypatch.setattr(main, "start_server", start_server)
    monkeypatch.setattr(main, "wait_ready", lambda *_a: steps.append("wait_ready"))
    monkeypatch.setattr(main, "stop_server", lambda _proc: steps.append("stop_server"))
    monkeypatch.setattr(main, "model_id", lambda _url: "m")
    monkeypatch.setattr(main, "repl", lambda *_a: steps.append("repl"))
    return steps


def test_main_runs_the_session_and_always_cleans_up(started: list[str]) -> None:
    """Build, start, talk, then remove the container and stop the server, even when something fails."""
    assert main.main(["llamahs", "fast"]) == 0
    assert started == [
        "ensure_image",
        "start_server",
        "wait_ready",
        "start_container",
        "repl",
        "remove_container",
        "stop_server",
    ]


def test_main_says_what_failed(
    started: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 2 for wrong arguments, 1 with the reason for a failure, 130 on Ctrl+C, cleaning up each time."""
    assert main.main(["llamahs"]) == USAGE
    assert "usage:" in capsys.readouterr().err

    def refused(*_a: object) -> None:
        raise subprocess.CalledProcessError(1, "docker run")

    monkeypatch.setattr(main, "start_container", refused)
    assert main.main(["llamahs", "fast"]) == FAILED
    assert "error:" in capsys.readouterr().err
    assert started[-2:] == ["remove_container", "stop_server"]

    def interrupted(*_a: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "wait_ready", interrupted)
    assert main.main(["llamahs", "fast"]) == INTERRUPTED
