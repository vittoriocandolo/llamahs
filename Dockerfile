FROM ubuntu

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
