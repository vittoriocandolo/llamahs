# Ubuntu 24.04, pinned by digest so that every build starts from the same image; Dependabot proposes updates.
FROM ubuntu:24.04@sha256:9c7f200c07e3d4a03e7249d2c16bf66a51120fb876aa2242cf31451412543289

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
