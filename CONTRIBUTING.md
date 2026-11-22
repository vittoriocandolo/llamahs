# Contributing

Thank you for helping with llamahs.

Report a bug or propose an idea in the [issues](https://github.com/vittoriocandolo/llamahs/issues/new/choose), where a
form asks for what is needed, and ask a question in the
[discussions](https://github.com/vittoriocandolo/llamahs/discussions). Report a vulnerability privately, as
[SECURITY.md](SECURITY.md) explains.

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
git ls-files -z '*.md' | xargs -0 uv run mdformat --check
uv run zizmor --persona auditor .
```

Coverage runs the tests and requires every line and every branch of `src/` to run, ruff runs every stable rule, mypy
checks all the code strictly, as Python 3.12 would run it, pytest turns any warning into an error, typos checks the
spelling of code and documentation, in British English with -ize (Oxford spelling: the -ize words in use are listed with
typos's settings in `pyproject.toml`), mdformat checks that the Markdown is formatted, wrapped at 120 as the code is,
and zizmor audits the GitHub workflows. `uv run pytest` alone runs the tests faster.
`git ls-files -z '*.md' | xargs -0 uv run mdformat` formats the Markdown. The tests stand in for Docker and
`llama-server`: a small local HTTP server answers as the server would.

CI runs mypy and the tests on Linux, macOS and Windows with every Python llamahs supports, and builds the image.

## Rules

- **The standard library only.** The harness runs as `python src/main.py`, with nothing to install.
- **The container stays locked down.** No network, a read-only root, no capabilities, no new privileges, only
  `/workspace` kept, as the caller's own user. A change that loosens any of these needs a reason in its pull request.
- **Defaults live in `src/settings.py`**, each with what it is for.
- **Every behaviour has a test**, and coverage stays at every line and every branch.
