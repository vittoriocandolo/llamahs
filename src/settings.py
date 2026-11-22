# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: MIT

"""The harness's defaults: the container, the server, the conversation and the tool."""

# The container the model works in, and the image it runs. Its limits keep a runaway command from taking the host
# with it: a fork bomb stops at the process limit, and a command filling memory or /tmp stops at theirs.
CONTAINER = "llamahs"
IMAGE = "llamahs:latest"
CONTAINER_MEMORY = "4g"  # the most memory the container's processes may hold, /tmp included
CONTAINER_PIDS = 512  # the most processes and threads at once
CONTAINER_TMP_SIZE = "1g"  # the most /tmp may hold
IMAGE_SOURCE_HASH_LABEL = "llamahs.source-hash"  # the hash of the Dockerfile the image was built from

# The server: where its log goes, how long it may take to come up, and how long it is given to stop.
SERVER_LOG_DIR = "logs"
SERVER_LOG_PREFIX = "llama-server"
DEFAULT_PORT = 8080  # when the profile's command names no --port
READY_TIMEOUT_SEC = 300
READY_POLL_SEC = 3
READY_REQUEST_TIMEOUT_SEC = 3
STOP_TIMEOUT_SEC = 5

# Requests to the server.
DEFAULT_MODEL_ID = "local-model"  # when the server names no model
CHAT_TIMEOUT_SEC = 3600  # an answer may take long on a slow machine
REQUEST_TIMEOUT_SEC = 60
SERVER_QUERY_TIMEOUT_SEC = 3  # the server's own facts: its model, its context, a token count
OUTPUT_MAX_TOKENS = 16384

# The conversation: how many tool rounds one message may take, and where its compacted context is kept.
MAX_TOOL_ROUNDS = 3
COMPACT_PROMPT_FILE = "prompts/compact.md"
SESSION_CONTEXT_PATH = "/tmp/CONTEXT.md"
SUMMARY_MAX_TOKENS = 8192

# The tool: how long a command may run, at most, and how much of its output the model sees.
TOOL_TIMEOUT_SEC = 60
TOOL_TIMEOUT_MAX = 600
TOOL_KILL_GRACE_SEC = 5  # after its time, a command that ignores the stop signal is killed this much later
TOOL_RESULT_CHARS = 8192
