# Changelog

All notable changes to llamahs are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- A harness that runs a `llama-server` command from a profile and gives its model one tool, `bash_exec`, in an Ubuntu
  container with no network, a read-only root, no capabilities, a persistent `/workspace` and a scratch `/tmp`, as the
  caller's own user.
- Commands stopped inside the container when they outlive their timeout, and their output cut to its start and end when
  it is long.
- Automatic context compaction into `/tmp/CONTEXT.md` as the context nears its end, and `/compact` on request.
- A failed request that ends only its turn, the conversation kept as it was before it.
- The image built only when the Dockerfile changes, from a base image pinned by digest.
- Exit statuses: 0 when the user leaves, 1 for a failure, 2 for wrong arguments, 130 after Ctrl+C.
