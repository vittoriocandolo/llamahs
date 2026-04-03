from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

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


def load_command(root: Path, name: str) -> str:
    path = root / "profiles" / name
    if not path.is_file():
        raise FileNotFoundError(f"missing profile: {path}")
    command = path.read_text(encoding="utf-8").strip()
    if not command:
        raise ValueError(f"empty profile: {path}")
    return command


def base_url(command: str) -> str:
    match = re.search(r"(?:^|\s)--port(?:=|\s+)(\d+)(?:\s|$)", command)
    port = match.group(1) if match is not None else str(DEFAULT_PORT)
    return f"http://127.0.0.1:{port}/v1"


def container_user() -> str | None:
    if os.name != "nt" and hasattr(os, "getuid") and hasattr(os, "getgid"):
        return f"{os.getuid()}:{os.getgid()}"
    return None


def server_log_path(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return root / SERVER_LOG_DIR / f"{SERVER_LOG_PREFIX}-{stamp}.log"


def docker(
    *args: str, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], check=check, text=True, **kwargs)


def read_container_text(path: str) -> str:
    args = ["exec", "-i"]
    user = container_user()
    if user is not None:
        args.extend(["-u", user])
    args.extend(
        [
            CONTAINER,
            "python3",
            "-c",
            (
                "from pathlib import Path; import sys; "
                "p = Path(sys.argv[1]); "
                "sys.stdout.write(p.read_text(encoding='utf-8') if p.is_file() else '')"
            ),
            path,
        ]
    )
    result = docker(*args, check=False, capture_output=True)
    return result.stdout if result.returncode == 0 else ""


def write_container_text(path: str, content: str) -> None:
    args = ["exec", "-i"]
    user = container_user()
    if user is not None:
        args.extend(["-u", user])
    args.extend(
        [
            CONTAINER,
            "python3",
            "-c",
            (
                "from pathlib import Path; import sys; "
                "Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')"
            ),
            path,
        ]
    )
    docker(*args, input=content)


def image_source_hash(root: Path) -> str:
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
    result = docker("image", "inspect", image, check=False, capture_output=True)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    config = payload[0].get("Config")
    if not isinstance(config, dict):
        return None
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        return None
    value = labels.get(name)
    return value if isinstance(value, str) else None


def ensure_image(root: Path) -> None:
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
    docker(
        "rm",
        "-f",
        CONTAINER,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_container(root: Path) -> None:
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
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        if os.name == "nt":
            return subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return subprocess.Popen(
            ["/bin/bash", "-lc", command],
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def wait_ready(proc: subprocess.Popen[str], url: str, log_path: Path) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SEC
    models_url = f"{url}/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited before becoming ready; see {log_path}"
            )
        try:
            with urllib.request.urlopen(
                models_url, timeout=READY_REQUEST_TIMEOUT_SEC
            ) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(READY_POLL_SEC)
    raise TimeoutError(
        f"llama-server did not become ready within {READY_TIMEOUT_SEC}s; see {log_path}"
    )


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=STOP_TIMEOUT_SEC)
