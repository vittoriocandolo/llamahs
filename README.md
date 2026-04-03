# llamahs

A basic harness designed to augment model capabilities with minimal inference overhead.

The model gets access to an Ubuntu Docker container with persistent `/workspace`, scratch `/tmp` and a read-only root filesystem.

No network access is provided.

`workspace/AGENTS.md` is available to the model by default.

Context compaction is automatic and based on `prompts/compact.md`. Manual `/compact` is also supported.

## Usage

From this directory copy `profiles/command.example` to `profiles/<name>` and put your `llama-server` command there.

Run `src/main.py <name>` with Python (Docker 20.10+ and Python 3.10+ are required).

`src/settings.py` contains the harness defaults if you want to review or change them.
