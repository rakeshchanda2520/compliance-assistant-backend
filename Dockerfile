# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Regrock backend
#
# Two stages, for one reason worth stating: wheels are built once in `build`
# and only the installed packages are copied into the final image, so pip's
# cache, build tooling and source archives never ship. The runtime image
# carries the interpreter, the dependencies, the code and the corpus.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS build

# Byte-compiled files and pip's version banner add noise to every layer and
# every log line. Off everywhere, so the two stages agree.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Requirements first, and ONLY requirements. Docker caches this layer on the
# file's hash, so editing a Python file does not reinstall numpy — the single
# biggest difference between a 4-second rebuild and a 3-minute one.
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt


# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# curl is here ONLY for HEALTHCHECK below. Without it the healthcheck cannot
# run and Docker reports the container unhealthy forever — a confusing way to
# discover a missing 3 MB package.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

# Never root. A container escape should not begin with a process that already
# owns the filesystem.
RUN useradd --create-home --uid 10001 regrock
WORKDIR /app

# The corpus is COPIED IN, not generated at build time and not mounted.
#
# This is the one thing about this image that is easy to get wrong. It was
# tried the other way — build the corpus during `docker build`, or mount it
# from the host — and both fail on any host without a shared filesystem, with
# `RuntimeError: search index missing at chunks.json` on every single boot.
#
# `data/chunks.json` and `data/embeddings.npz` are real committed files,
# produced offline by KG_creation/ and shipped as part of the image. Updating
# the law means rebuilding the image, which is correct: the corpus IS a
# version of this service, and `build_id` (a content hash) makes that visible
# in every answer.
COPY --chown=regrock:regrock . .

USER regrock

EXPOSE 8100

# LIVENESS, not readiness — and the difference matters here.
#
# `/api/live` touches no dependency. `/api/health` returns 503 when Neo4j or
# the model provider is unreachable, which is exactly right for a load
# balancer (stop sending traffic) and exactly wrong for Docker (restart the
# container). Restarting this process because Neo4j blinked fixes nothing and
# throws away a warm corpus; the service is designed to degrade instead.
#
# start-period is 40s because startup loads the corpus and can spend ~10s
# waiting out a Neo4j DNS timeout before falling back.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8100/api/live || exit 1

# Two workers by default. Each holds its own copy of the corpus and the dense
# matrix (~30 MB resident), so this trades memory for concurrency — raise it
# with WEB_CONCURRENCY once you know the container's memory limit.
ENV WEB_CONCURRENCY=2

# A shell is needed to expand ${WEB_CONCURRENCY}, and `exec` is what makes
# this safe: without it the shell stays PID 1, and /bin/sh does not forward
# SIGTERM to its children. Every deploy would then hang until Docker's
# 10-second timeout and SIGKILL the workers — cutting SSE responses off
# mid-answer. With `exec`, the shell REPLACES itself with uvicorn, so uvicorn
# is PID 1 and shuts down gracefully.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port 8100 --workers ${WEB_CONCURRENCY}"]
