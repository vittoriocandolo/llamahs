# AGENTS.md

## Runtime

- `/workspace` is the persistent shared directory.
- `/tmp` is scratch space for temporary work.
- The container root filesystem is read-only.
- Outbound network access is disabled.

## Guidelines

- If the environment blocks your work, stop and ask the user what to do next.
- Prefer `rg`, `sed -n`, `head`, `tail`, and `jq` for direct text and JSON work.
- Prefer direct shell tools and short `python3` scripts over heavier workflows.
