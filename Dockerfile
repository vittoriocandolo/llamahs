# Ubuntu 26.04, pinned by digest so that every build starts from the same image; Dependabot proposes updates.
FROM ubuntu:26.04@sha256:d66b366dac61b575bee2aef07e9c26e8070096fbe84a3d4d3eb0da069b5d76db

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
