# llamahs

A basic harness designed to augment model capabilities with minimal inference overhead.

The model gets access to an Ubuntu Docker container with persistent `/workspace`, scratch `/tmp` and a read-only root filesystem.

No network access is provided.

`workspace/AGENTS.md` is available to the model by default.

Context compaction is automatic and based on `prompts/compact.md`. Manual `/compact` is also supported.

## Usage

From this directory copy `profiles/command.example` to `profiles/<name>` and put your `llama-server` command there.

Run `src/main.py <name>` with Python (Docker 20.10+ and Python 3.12+ are required).

`src/settings.py` contains the harness defaults if you want to review or change them.

A command the model runs is stopped inside the container once it outlives its timeout (60 s by default, at most
600 s). A request to the server that fails ends only that turn: the conversation goes back to before it, and the
session goes on.

## Development

The harness uses the standard library only. The checks run with [uv](https://docs.astral.sh/uv/):

```
uv run ruff check
uv run ruff format --check
uv run mypy
uv run coverage run -m pytest
uv run coverage report
```

The tests stand in for Docker and `llama-server`, so they need neither.
