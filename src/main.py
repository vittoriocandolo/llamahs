# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The harness: start the server and the container, then talk with the model, which works through one bash tool."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

from runtime import (
    base_url,
    ensure_image,
    load_command,
    read_container_text,
    remove_container,
    server_log_path,
    start_container,
    start_server,
    stop_server,
    wait_ready,
    write_container_text,
)
from settings import (
    CHAT_TIMEOUT_SEC,
    COMPACT_PROMPT_FILE,
    DEFAULT_MODEL_ID,
    MAX_TOOL_ROUNDS,
    OUTPUT_MAX_TOKENS,
    REQUEST_TIMEOUT_SEC,
    SERVER_QUERY_TIMEOUT_SEC,
    SESSION_CONTEXT_PATH,
    SUMMARY_MAX_TOKENS,
)
from tool import TOOLS, as_count, parse_args, run_tool_call

type Message = dict[str, object]


class Session(NamedTuple):
    """Where the conversation happens: the harness's folder, the server's address, its model and its context size."""

    root: Path
    url: str
    model: str
    n_ctx: int | None


# What a request to the server can fail with: no answer or a refusal (OSError, which urllib's errors and timeouts
# are), an error status (RuntimeError, as request_json raises it), or an answer that is not what was asked for
# (ValueError, which a JSON decoding error is too, or TypeError, for an answer of the wrong shape).
REQUEST_FAILURES = (OSError, RuntimeError, ValueError, TypeError)


def request_json(url: str, payload: object = None, timeout: float = REQUEST_TIMEOUT_SEC) -> object:
    """Send a request to the server, a POST of `payload` as JSON or a GET without one, and return its JSON answer."""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        with exc:  # its connection is let go once its body is read
            body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(body or f"http {exc.code}") from exc


def message_of(response: object) -> Message:
    """Return the message of a chat answer, or refuse an answer that has none."""
    choices = response.get("choices") if isinstance(response, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        msg = f"the server's answer has no message: {str(response)[:200]}"
        raise TypeError(msg)
    return {str(k): v for k, v in message.items()}


def text_of(value: object) -> str:
    """Return a message's text: its content if it is a string, else nothing."""
    return value if isinstance(value, str) else ""


def model_id(url: str) -> str:
    """Return the name of the model the server runs, or the default if it does not say."""
    payload = request_json(f"{url}/models", timeout=SERVER_QUERY_TIMEOUT_SEC)
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, list) and data and isinstance(data[0], dict):
        value = data[0].get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(payload, dict):
        value = payload.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_MODEL_ID


def chat(url: str, model: str, messages: list[Message]) -> object:
    """Ask the model for its next message, with the tool it may call."""
    return request_json(
        f"{url}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": OUTPUT_MAX_TOKENS,
        },
        timeout=CHAT_TIMEOUT_SEC,
    )


def context_limit(url: str) -> int | None:
    """Return the server's context size in tokens, from its properties or its slots, or None if it does not say."""
    try:
        payload = request_json(f"{url}/props", timeout=SERVER_QUERY_TIMEOUT_SEC)
    except REQUEST_FAILURES:
        payload = None
    if isinstance(payload, dict):
        settings = payload.get("default_generation_settings")
        if isinstance(settings, dict):
            value = as_count(settings.get("n_ctx"), 1)
            if value is not None:
                return value
    try:
        payload = request_json(f"{url}/slots", timeout=SERVER_QUERY_TIMEOUT_SEC)
    except REQUEST_FAILURES:
        return None
    if isinstance(payload, list):
        for slot in payload:
            if not isinstance(slot, dict):
                continue
            value = as_count(slot.get("n_ctx"), 1)
            if value is not None:
                return value
    return None


def context_usage(response: object) -> int | None:
    """Return how many tokens of context an answer used, from its timings, or None if it does not say."""
    timings = response.get("timings") if isinstance(response, dict) else None
    if not isinstance(timings, dict):
        return None
    cache_n = as_count(timings.get("cache_n"))
    prompt_n = as_count(timings.get("prompt_n"))
    predicted_n = as_count(timings.get("predicted_n"))
    if cache_n is None or prompt_n is None or predicted_n is None:
        return None
    return cache_n + prompt_n + predicted_n


def compact_prompt_path(root: Path) -> Path:
    """Return the prompt that asks the model to compact the conversation."""
    return root / COMPACT_PROMPT_FILE


def context_note() -> str:
    """Return the note that starts a compacted conversation: where its context is kept."""
    return f"Context is available in {SESSION_CONTEXT_PATH}. Read it with bash_exec if needed."


def seed_messages() -> list[Message]:
    """Return the start of a conversation: the context note if there is compacted context, else nothing."""
    if read_container_text(SESSION_CONTEXT_PATH).strip():
        return [{"role": "assistant", "content": context_note()}]
    return []


def summarize_history(session: Session, messages: list[Message]) -> str:
    """Ask the model to compact the conversation, merged with any context compacted before it."""
    prior = read_container_text(SESSION_CONTEXT_PATH).strip()
    prior_block = ""
    if prior:
        prior_block = f"\nMerge and refresh this existing {SESSION_CONTEXT_PATH} if still relevant:\n\n{prior}\n"
    template = compact_prompt_path(session.root).read_text(encoding="utf-8")
    prompt = template.format(context_path=SESSION_CONTEXT_PATH, prior_block=prior_block)
    response = request_json(
        f"{session.url}/chat/completions",
        {
            "model": session.model,
            "messages": [*messages, {"role": "user", "content": prompt}],
            "max_tokens": min(OUTPUT_MAX_TOKENS, SUMMARY_MAX_TOKENS),
        },
        timeout=CHAT_TIMEOUT_SEC,
    )
    summary = text_of(message_of(response).get("content")).strip()
    return summary or prior or "# Context\n\nNo durable context yet."


def token_count(url: str, content: str) -> int | None:
    """Return how many tokens a text is, by the server's tokenizer, or None if it cannot say."""
    try:
        tokenized = request_json(f"{url}/tokenize", {"content": content}, timeout=SERVER_QUERY_TIMEOUT_SEC)
    except REQUEST_FAILURES:
        return None
    tokens = tokenized.get("tokens") if isinstance(tokenized, dict) else None
    return len(tokens) if isinstance(tokens, list) else None


def prompt_tokens(url: str, messages: list[Message]) -> int | None:
    """Return how many tokens the conversation and the tool's description take, or None if the server cannot say."""
    try:
        templated = request_json(f"{url}/apply-template", {"messages": messages}, timeout=SERVER_QUERY_TIMEOUT_SEC)
    except REQUEST_FAILURES:
        return None
    prompt = templated.get("prompt") if isinstance(templated, dict) else None
    if not isinstance(prompt, str):
        return None
    prompt_count = token_count(url, prompt)
    if prompt_count is None:
        return None
    tool_count = token_count(url, json.dumps(TOOLS, separators=(",", ":"), ensure_ascii=False))
    if tool_count is None:
        return None
    return prompt_count + tool_count


def maybe_compact_history(
    session: Session,
    messages: list[Message],
    used_tokens: int | None,
    *,
    force: bool = False,
) -> list[Message]:
    """Compact the conversation once it nears the context's end, or when asked; return what the conversation is now.

    The context must keep room for the next answer and for the summary itself. A compaction that fails leaves the
    conversation as it was.
    """
    if not messages:
        print("[nothing to compact]")
        return messages
    if used_tokens is None and not force:
        used_tokens = prompt_tokens(session.url, messages)
    room = OUTPUT_MAX_TOKENS + SUMMARY_MAX_TOKENS
    if not force and (session.n_ctx is None or used_tokens is None or used_tokens + room < session.n_ctx):
        return messages
    try:
        write_container_text(SESSION_CONTEXT_PATH, summarize_history(session, messages).rstrip() + "\n")
    except (*REQUEST_FAILURES, KeyError, IndexError, subprocess.SubprocessError) as exc:
        # KeyError or IndexError: a prompt naming a field the template does not fill; SubprocessError: the container
        # would not write
        print(f"[context compaction skipped: {exc}]")
        return messages
    print(f"[context compacted to {SESSION_CONTEXT_PATH}]")
    return seed_messages()


COMMANDS = "commands: exit | quit | compact"


def repl(root: Path, url: str, model: str) -> None:
    """Talk with the model until the user leaves; a turn whose request fails is dropped, and the session goes on."""
    session = Session(root, url, model, context_limit(url))
    messages = seed_messages()
    print(f"ready model={model} url={url} tool=bash_exec workspace=/workspace")
    print(COMMANDS)
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return
        if prompt.lower() in {"compact", "/compact"}:
            messages = maybe_compact_history(session, messages, None, force=True)
            continue
        turn_start = len(messages)
        messages.append({"role": "user", "content": prompt})
        try:
            messages = take_turn(session, messages)
        except REQUEST_FAILURES as exc:
            # a failed request ends this turn only: the history goes back to before it, and the session goes on
            del messages[turn_start:]
            print(f"[request failed, nothing kept from this turn: {exc}]")


def tool_call_message(content: str, calls: list[dict[str, object]]) -> Message:
    """Return the model's message that asks for tools, its calls' arguments written as compact JSON."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": function_of(call).get("name"),
                    "arguments": json.dumps(parse_args(function_of(call).get("arguments")), separators=(",", ":")),
                },
            }
            for call in calls
        ],
    }


def function_of(call: dict[str, object]) -> dict[str, object]:
    """Return the function a tool call names, or refuse a call that names none."""
    function = call.get("function")
    if not isinstance(function, dict) or "id" not in call:
        msg = f"a tool call without an id or a function: {str(call)[:200]}"
        raise TypeError(msg)
    return {str(k): v for k, v in function.items()}


def take_turn(session: Session, messages: list[Message]) -> list[Message]:
    """Answer the user's last message, running the tools it asks for, at most MAX_TOOL_ROUNDS times."""
    used_tokens = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = chat(session.url, session.model, messages)
        used_tokens = context_usage(response)
        msg = message_of(response)
        raw_calls = msg.get("tool_calls") or []
        calls = [c for c in raw_calls if isinstance(c, dict)] if isinstance(raw_calls, list) else []
        for call in calls:
            function_of(call)  # every call is checked before any of them runs
        text = text_of(msg.get("content")).strip()
        if not calls:
            answer = text or "(no response)"
            messages.append({"role": "assistant", "content": answer})
            print(f"assistant> {answer}")
            return maybe_compact_history(session, messages, used_tokens)
        messages.append(tool_call_message(text_of(msg.get("content")), calls))
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": function_of(call).get("name"),
                "content": run_tool_call(call),
            }
            for call in calls
        )
    answer = "Stopped after max tool rounds."
    messages.append({"role": "assistant", "content": answer})
    print(f"assistant> {answer}")
    return maybe_compact_history(session, messages, used_tokens)


ARGUMENTS = 2  # the program and its profile


def main(argv: list[str]) -> int:
    """Run the harness with a profile: 0 when the user leaves, 1 if it fails, 2 for wrong arguments."""
    if len(argv) != ARGUMENTS:
        print(f"usage: {argv[0]} <profile>", file=sys.stderr)
        return 2
    try:
        root = Path(__file__).resolve().parents[1]
        command = load_command(root, argv[1])
        url = base_url(command)
        ensure_image(root)
        log_path = server_log_path(root)
        proc = start_server(log_path, command)
        try:
            wait_ready(proc, url, log_path)
            start_container(root)
            repl(root, url, model_id(url))
            return 0
        finally:
            try:
                remove_container()
            finally:
                stop_server(proc)  # whatever became of the container
    except KeyboardInterrupt:  # Ctrl+C while starting: stopped quietly, with the status a shell expects
        return 130
    except (*REQUEST_FAILURES, subprocess.SubprocessError) as exc:  # SubprocessError: Docker refused a command
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
