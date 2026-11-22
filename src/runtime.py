# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The runtime: the profile's server command, the server's process and log, and the container the model works in."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from http import HTTPStatus
from typing import IO, TYPE_CHECKING, TypedDict, Unpack

from settings import (
    CONTAINER,
    DEFAULT_PORT,
    IMAGE,
    IMAGE_SOURCE_HASH_LABEL,
    READY_POLL_SEC,
    READY_REQUEST_TIMEOUT_SEC,
    READY_TIMEOUT_SEC,
    SERVER_LOG_DIR,
    SERVER_LOG_PREFIX,
    STOP_TIMEOUT_SEC,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_command(root: Path, name: str) -> str:
    """Return the server command a profile holds, refusing a profile that is missing or empty."""
    path = root / "profiles" / name
    if not path.is_file():
        msg = f"missing profile: {path}"
        raise FileNotFoundError(msg)
    command = path.read_text(encoding="utf-8").strip()
    if not command:
        msg = f"empty profile: {path}"
        raise ValueError(msg)
    return command


def base_url(command: str) -> str:
    """Return the server's API address: on this machine, at the command's --port, or the default one."""
    match = re.search(r"(?:^|\s)--port(?:=|\s+)(\d+)(?:\s|$)", command)
    port = match.group(1) if match is not None else str(DEFAULT_PORT)
    return f"http://127.0.0.1:{port}/v1"


def container_user() -> str | None:
    """Return the user and group to run as in the container, the caller's own, so files it writes stay theirs.

    None on Windows, where Docker maps the files itself.
    """
    user = None
    if sys.platform != "win32":
        user = f"{os.getuid()}:{os.getgid()}"
    return user


def server_log_path(root: Path) -> Path:
    """Return a new log file for the server, named by the local time it starts."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    return root / SERVER_LOG_DIR / f"{SERVER_LOG_PREFIX}-{stamp}.log"


class RunOptions(TypedDict, total=False):
    """The parts of subprocess.run that the harness passes on to Docker: `check` is True unless given."""

    check: bool
    capture_output: bool
    input: str
    timeout: float
    stdout: int
    stderr: int


def docker(*args: str, **options: Unpack[RunOptions]) -> subprocess.CompletedProcess[str]:
    """Run a Docker command, as text, checked unless `check` is False; FileNotFoundError if Docker is not installed."""
    executable = shutil.which("docker")
    if executable is None:
        msg = "docker: not found; the harness needs Docker 20.10 or later"
        raise FileNotFoundError(msg)
    return subprocess.run(
        [executable, *args],
        check=options.get("check", True),
        text=True,
        capture_output=options.get("capture_output", False),
        input=options.get("input"),
        timeout=options.get("timeout"),
        stdout=options.get("stdout"),
        stderr=options.get("stderr"),
    )


def exec_args() -> list[str]:
    """Return the start of a `docker exec` into the container, as the caller's own user."""
    args = ["exec", "-i"]
    user = container_user()
    if user is not None:
        args.extend(["-u", user])
    return args


READ_FILE = (
    "from pathlib import Path; import sys; p = Path(sys.argv[1]); "
    "sys.stdout.write(p.read_text(encoding='utf-8') if p.is_file() else '')"
)
WRITE_FILE = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')"


def read_container_text(path: str) -> str:
    """Return a text file in the container, or nothing if it is not there or cannot be read."""
    result = docker(*exec_args(), CONTAINER, "python3", "-c", READ_FILE, path, check=False, capture_output=True)
    return result.stdout if result.returncode == 0 else ""


def write_container_text(path: str, content: str) -> None:
    """Write a text file in the container."""
    docker(*exec_args(), CONTAINER, "python3", "-c", WRITE_FILE, path, input=content)


def image_source_hash(root: Path) -> str:
    """Return the hash of what the image is built from: the Dockerfile and the .dockerignore."""
    hasher = hashlib.sha256()
    for rel in ("Dockerfile", ".dockerignore"):
        path = root / rel
        if not path.is_file():
            continue
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def image_label(image: str, name: str) -> str | None:
    """Return one label of an image, or None if the image or the label is not there."""
    result = docker("image", "inspect", image, check=False, capture_output=True)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return None
    config = payload[0].get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    value = labels.get(name) if isinstance(labels, dict) else None
    return value if isinstance(value, str) else None


def ensure_image(root: Path) -> None:
    """Build the image, unless the one there was built from these very files."""
    source_hash = image_source_hash(root)
    if image_label(IMAGE, IMAGE_SOURCE_HASH_LABEL) == source_hash:
        return
    docker(
        "build",
        "-f",
        str(root / "Dockerfile"),
        "--label",
        f"{IMAGE_SOURCE_HASH_LABEL}={source_hash}",
        "-t",
        IMAGE,
        str(root),
    )


def remove_container() -> None:
    """Remove the container, if there is one."""
    docker("rm", "-f", CONTAINER, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_container(root: Path) -> None:
    """Start the container afresh: no network, a read-only root, /workspace kept, /tmp scratch, no privileges."""
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    remove_container()
    args = [
        "run",
        "--rm",
        "-d",
        "--name",
        CONTAINER,
        "--read-only",
        "--network",
        "none",
        "--tmpfs",
        "/tmp:exec,mode=1777",
        "-e",
        "HOME=/tmp",
        "-e",
        "XDG_CACHE_HOME=/tmp/.cache",
        "-v",
        f"{workspace.resolve()}:/workspace",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "-w",
        "/workspace",
    ]
    user = container_user()
    if user is not None:
        args.extend(["--user", user])
    docker(*args, IMAGE, "tail", "-f", "/dev/null", stdout=subprocess.DEVNULL)


def start_server(path: Path, command: str) -> subprocess.Popen[str]:
    """Start the profile's server command, its output to a new log file."""
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        return launch(command, log)


def launch(command: str, log: IO[str]) -> subprocess.Popen[str]:
    """Run a command line as its shell would: cmd on Windows, a bash login shell elsewhere."""
    on_windows = sys.platform == "win32"
    return subprocess.Popen(
        command if on_windows else ["/bin/bash", "-lc", command],
        shell=on_windows,
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def wait_ready(proc: subprocess.Popen[str], url: str, log_path: Path) -> None:
    """Wait until the server answers, or give up if it exits or does not answer in time."""
    deadline = time.monotonic() + READY_TIMEOUT_SEC
    models_url = f"{url}/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            msg = f"llama-server exited before becoming ready; see {log_path}"
            raise RuntimeError(msg)
        try:
            with urllib.request.urlopen(models_url, timeout=READY_REQUEST_TIMEOUT_SEC) as response:
                if HTTPStatus.OK <= response.status < HTTPStatus.MULTIPLE_CHOICES:
                    return
        except urllib.error.HTTPError as exc:
            exc.close()  # an error status: up, but not ready yet; its connection is let go
        except (urllib.error.URLError, TimeoutError):
            pass  # not up yet: asked again after the pause
        time.sleep(READY_POLL_SEC)
    msg = f"llama-server did not become ready within {READY_TIMEOUT_SEC}s; see {log_path}"
    raise TimeoutError(msg)


def stop_server(proc: subprocess.Popen[str]) -> None:
    """Stop the server, killing it if it does not stop in time."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=STOP_TIMEOUT_SEC)
