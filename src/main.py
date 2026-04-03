from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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
from tool import TOOLS, parse_args, run_tool_call


def request_json(
    url: str, payload: Any = None, timeout: int = REQUEST_TIMEOUT_SEC
) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(body or f"http {exc.code}") from exc


def model_id(url: str) -> str:
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


def chat(url: str, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
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


def as_int(value: Any, minimum: int = 0) -> int | None:
    try:
        number = int(value)
    except Exception:
        return None
    return number if number >= minimum else None


def context_limit(url: str) -> int | None:
    try:
        payload = request_json(f"{url}/props", timeout=SERVER_QUERY_TIMEOUT_SEC)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        settings = payload.get("default_generation_settings")
        if isinstance(settings, dict):
            value = as_int(settings.get("n_ctx"), 1)
            if value is not None:
                return value
    try:
        payload = request_json(f"{url}/slots", timeout=SERVER_QUERY_TIMEOUT_SEC)
    except Exception:
        return None
    if isinstance(payload, list):
        for slot in payload:
            if not isinstance(slot, dict):
                continue
            value = as_int(slot.get("n_ctx"), 1)
            if value is not None:
                return value
    return None


def context_usage(response: dict[str, Any]) -> int | None:
    timings = response.get("timings")
    if not isinstance(timings, dict):
        return None
    cache_n = as_int(timings.get("cache_n"))
    prompt_n = as_int(timings.get("prompt_n"))
    predicted_n = as_int(timings.get("predicted_n"))
    if cache_n is None or prompt_n is None or predicted_n is None:
        return None
    return cache_n + prompt_n + predicted_n


def compact_prompt_path(root: Path) -> Path:
    return root / COMPACT_PROMPT_FILE


def context_note() -> str:
    return f"Context is available in {SESSION_CONTEXT_PATH}. Read it with bash_exec if needed."


def seed_messages() -> list[dict[str, Any]]:
    if read_container_text(SESSION_CONTEXT_PATH).strip():
        return [{"role": "assistant", "content": context_note()}]
    return []


def summarize_history(
    root: Path,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
) -> str:
    prior = read_container_text(SESSION_CONTEXT_PATH).strip()
    prior_block = ""
    if prior:
        prior_block = f"\nMerge and refresh this existing {SESSION_CONTEXT_PATH} if still relevant:\n\n{prior}\n"
    prompt = (
        compact_prompt_path(root)
        .read_text(encoding="utf-8")
        .format(
            context_path=SESSION_CONTEXT_PATH,
            prior_block=prior_block,
        )
    )
    response = request_json(
        f"{url}/chat/completions",
        {
            "model": model,
            "messages": [*messages, {"role": "user", "content": prompt}],
            "max_tokens": min(OUTPUT_MAX_TOKENS, SUMMARY_MAX_TOKENS),
        },
        timeout=CHAT_TIMEOUT_SEC,
    )
    content = response["choices"][0]["message"].get("content")
    summary = content.strip() if isinstance(content, str) else ""
    return summary or prior or "# Context\n\nNo durable context yet."


def token_count(url: str, content: str) -> int | None:
    try:
        tokenized = request_json(
            f"{url}/tokenize",
            {"content": content},
            timeout=SERVER_QUERY_TIMEOUT_SEC,
        )
    except Exception:
        return None
    if not isinstance(tokenized, dict):
        return None
    tokens = tokenized.get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


def prompt_tokens(url: str, messages: list[dict[str, Any]]) -> int | None:
    try:
        templated = request_json(
            f"{url}/apply-template",
            {"messages": messages},
            timeout=SERVER_QUERY_TIMEOUT_SEC,
        )
        prompt = templated.get("prompt") if isinstance(templated, dict) else None
        if not isinstance(prompt, str):
            return None
    except Exception:
        return None
    prompt_count = token_count(url, prompt)
    if prompt_count is None:
        return None
    tool_count = token_count(
        url,
        json.dumps(TOOLS, separators=(",", ":"), ensure_ascii=False),
    )
    if tool_count is None:
        return None
    return prompt_count + tool_count


def maybe_compact_history(
    root: Path,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    n_ctx: int | None,
    used_tokens: int | None,
    force: bool = False,
) -> list[dict[str, Any]]:
    if not messages:
        print("[nothing to compact]")
        return messages
    if used_tokens is None and not force:
        used_tokens = prompt_tokens(url, messages)
    if not force and (
        n_ctx is None
        or used_tokens is None
        or used_tokens + OUTPUT_MAX_TOKENS + SUMMARY_MAX_TOKENS < n_ctx
    ):
        return messages
    try:
        write_container_text(
            SESSION_CONTEXT_PATH,
            summarize_history(root, url, model, messages).rstrip() + "\n",
        )
        print(f"[context compacted to {SESSION_CONTEXT_PATH}]")
        return seed_messages()
    except Exception as exc:
        print(f"[context compaction skipped: {exc}]")
        return messages


def repl(root: Path, url: str, model: str) -> None:
    n_ctx = context_limit(url)
    messages = seed_messages()
    print(f"ready model={model} url={url} tool=bash_exec workspace=/workspace")
    print("commands: exit | quit | compact")
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
            messages = maybe_compact_history(
                root, url, model, messages, n_ctx, None, force=True
            )
            continue
        messages.append({"role": "user", "content": prompt})
        used_tokens = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = chat(url, model, messages)
            used_tokens = context_usage(response)
            msg = response["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            text = (msg.get("content") or "").strip()
            if not calls:
                answer = text or "(no response)"
                messages.append({"role": "assistant", "content": answer})
                print(f"assistant> {answer}")
                messages = maybe_compact_history(
                    root, url, model, messages, n_ctx, used_tokens
                )
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": json.dumps(
                                    parse_args(call["function"].get("arguments")),
                                    separators=(",", ":"),
                                ),
                            },
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                output = run_tool_call(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                        "content": output,
                    }
                )
        else:
            answer = "Stopped after max tool rounds."
            messages.append({"role": "assistant", "content": answer})
            print(f"assistant> {answer}")
            messages = maybe_compact_history(
                root, url, model, messages, n_ctx, used_tokens
            )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
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
            remove_container()
            stop_server(proc)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
