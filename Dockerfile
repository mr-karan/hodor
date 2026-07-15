FROM oven/bun:1.3.14@sha256:e10577f0db68676a7024391c6e5cb4b879ebd17188ab750cf10024a6d700e5c4 AS base
WORKDIR /build

# Install dependencies into temp dirs for caching
FROM base AS install

# Dev dependencies (for build)
RUN mkdir -p /temp/dev
COPY package.json bun.lock /temp/dev/
RUN cd /temp/dev && bun install --frozen-lockfile

# Production dependencies only
RUN mkdir -p /temp/prod
COPY package.json bun.lock /temp/prod/
RUN cd /temp/prod && bun install --frozen-lockfile --production

# Build stage
FROM base AS build
COPY --from=install /temp/dev/node_modules node_modules
COPY tsconfig.json tsup.config.ts package.json ./
COPY src ./src
COPY templates ./templates
RUN bun run build

# Final stage
FROM oven/bun:1.3.14-slim@sha256:d56a2534ffd262e92c12fd3249d3924d296d97086da773f821d7d0477435ea04

# Install system dependencies for PR reviews
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        jq \
        tree \
        less \
        python3 \
        shellcheck \
        file \
        diffstat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install ripgrep
RUN curl -fsSL "https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep_15.1.0-1_amd64.deb" -o /tmp/ripgrep.deb && \
    echo "62ce186675c71edbc7c48a67dc36f986974551354d6ce63726adc2d85aad802c  /tmp/ripgrep.deb" | sha256sum -c - && \
    dpkg -i /tmp/ripgrep.deb && \
    rm /tmp/ripgrep.deb

# Install GitHub CLI (gh)
RUN curl -fsSL "https://github.com/cli/cli/releases/download/v2.83.0/gh_2.83.0_linux_amd64.tar.gz" -o /tmp/gh.tar.gz && \
    echo "a5cf6cdb40fc67751adf561126b3314044779cea81ba4f254fbe8e9a69f1676f  /tmp/gh.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/gh.tar.gz -C /tmp && \
    mv /tmp/gh_2.83.0_linux_amd64/bin/gh /usr/local/bin/ && \
    rm -rf /tmp/gh*

# Install GitLab CLI (glab)
RUN curl -fsSL "https://gitlab.com/gitlab-org/cli/-/releases/v1.93.0/downloads/glab_1.93.0_linux_amd64.tar.gz" -o /tmp/glab.tar.gz && \
    echo "300f3c12bd75f298747364f382f978bbe63809ef660bb2969925f343f9c20ae4  /tmp/glab.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/glab.tar.gz -C /tmp && \
    mv /tmp/bin/glab /usr/local/bin/ && \
    rm -rf /tmp/glab* /tmp/bin

WORKDIR /app

# Copy built application and production deps
COPY --chown=bun:bun --from=install /temp/prod/node_modules node_modules
COPY --chown=bun:bun --from=build /build/dist ./dist
COPY --chown=bun:bun --from=build /build/templates ./templates
COPY --chown=bun:bun --from=build /build/package.json ./

# Set wider terminal dimensions
ENV COLUMNS=200
ENV LINES=50

# Workspace for cloned repos
RUN install -d -o bun -g bun /workspace /tmp/hodor

LABEL org.opencontainers.image.title="Hodor" \
      org.opencontainers.image.description="AI-powered code review agent for GitHub, GitLab, and Gitea/Forgejo" \
      org.opencontainers.image.url="https://github.com/mr-karan/hodor" \
      org.opencontainers.image.source="https://github.com/mr-karan/hodor" \
      org.opencontainers.image.vendor="Karan Sharma" \
      org.opencontainers.image.licenses="MIT"

USER bun

ENV HODOR_WORKSPACE=/workspace

ENTRYPOINT ["bun", "run", "dist/cli.js"]
CMD ["--help"]
