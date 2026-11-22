# Ubuntu 24.04, pinned by digest so that every build starts from the same image; Dependabot proposes updates.
FROM ubuntu:25.10@sha256:7cc5e35f6567ee8c66d2abb4aab0fd866669e6207c237c3a8f0947a5c7f17092

RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo Etc/UTC >/etc/timezone \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    jq \
    python3 \
    ripgrep \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
