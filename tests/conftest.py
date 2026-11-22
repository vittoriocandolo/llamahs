# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""Shared fixtures: a small HTTP server that answers as llama-server would, on this machine."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar, override

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

type Handler = Callable[[str, object], tuple[int, object]]


class Script:
    """What the server answers, and the requests it has seen: set by each test."""

    handler: Handler = staticmethod(lambda _path, _body: (404, {}))
    seen: ClassVar[list[tuple[str, str, object]]] = []


class Answers(BaseHTTPRequestHandler):
    """Answer each request with what the test's handler returns for its path and its JSON body."""

    def answer(self, body: object) -> None:
        """Answer a request with the handler's status and JSON."""
        Script.seen.append((self.command, self.path, body))
        status, payload = Script.handler(self.path, body)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        """Answer a GET."""
        self.answer(None)

    def do_POST(self) -> None:
        """Answer a POST, its body read as JSON."""
        length = int(self.headers.get("Content-Length", "0"))
        self.answer(json.loads(self.rfile.read(length)))

    @override
    def log_message(self, format: str, *args: object) -> None:
        """Say nothing: the tests read what was asked from `seen`."""


class Server:
    """A local server whose answers a test sets, and the requests it has seen."""

    def __init__(self, httpd: ThreadingHTTPServer) -> None:
        """Serve on the given server."""
        self.httpd = httpd
        self.url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"

    def answers(self, handler: Handler) -> None:
        """Answer every request with `handler(path, body)`: a status and a JSON payload (or raw bytes)."""
        Script.handler = handler

    @property
    def seen(self) -> list[tuple[str, str, object]]:
        """Return the requests so far: method, path and JSON body."""
        return Script.seen


@pytest.fixture
def server() -> Iterator[Server]:
    """Run a local server for one test."""
    Script.seen = []
    Script.handler = lambda _path, _body: (404, {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Answers)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield Server(httpd)
    finally:
        httpd.shutdown()
        httpd.server_close()
