# Security

llamahs runs a local `llama-server` command from a profile and lets its model run bash commands in a Docker container
with no network, a read-only root, no capabilities and only `/workspace` kept. If you find a way out of that container,
or anything else that could do harm (in the harness, its image, or the workflows that check it), please report it
privately rather than in a public issue.

**How:** [report it privately](https://github.com/vittoriocandolo/llamahs/security/advisories/new) (GitHub's private
vulnerability reporting: the repository's Security tab, "Report a vulnerability"). Say what you found, how to see it,
and what it could do. You will get an answer within a week.

Only the latest version, on `main`, is looked after.
