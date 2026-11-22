# Contributing

Thank you for helping with llamahs.

## Set up

llamahs uses [uv](https://docs.astral.sh/uv/) for its development tools; the harness itself uses the standard library
only:

```
uv sync
```

## Check a change

```
uv run coverage run -m pytest
uv run coverage report
uv run ruff check
uv run ruff format --check
uv run mypy
uv run typos
uv run zizmor --persona auditor .
```

Coverage runs the tests and requires every line and every branch of `src/` to run, ruff runs every stable rule, mypy
checks all the code strictly, as Python 3.12 would run it, pytest turns any warning into an error, typos checks the
spelling of code and documentation, and zizmor audits the GitHub workflows. `uv run pytest` alone runs the tests
faster. The tests stand in for Docker and `llama-server`: a small local HTTP server answers as the server would.

CI runs mypy and the tests on Linux, macOS and Windows with every Python llamahs supports, and builds the image.

## Rules

- **The standard library only.** The harness runs as `python src/main.py`, with nothing to install.
- **The container stays locked down.** No network, a read-only root, no capabilities, no new privileges, only
  `/workspace` kept, as the caller's own user. A change that loosens any of these needs a reason in its pull request.
- **Defaults live in `src/settings.py`**, each with what it is for.
- **Every behavior has a test**, and coverage stays at every line and every branch.
