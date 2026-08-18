# Standalone backend service — this repo (backend/) deploys independently
# of kg_build/ and the frontend, which live in a separate checkout/repo.
# `python -m kg_build` (the corpus builder) is NOT run from here and never
# has been part of this image; it's an offline step, run from the monorepo
# checkout that still has kg_build/ and the source PDFs, producing
# data/chunks.json and pushing to Neo4j. This image only ever SERVES that
# already-built corpus — it has no way to build one itself, by design: a
# serving container that could also rebuild the graph is a container that
# could serve a half-built one.
#
# This image serves no HTML at all (app.py has no "/" route); it answers
# /api/* only, and expects DPDP_CORS_ORIGINS to name wherever the frontend
# is actually deployed, or the browser blocks every request from reaching it.
#
# data/chunks.json and data/vocab.yaml are BAKED INTO THIS IMAGE (see the
# COPY below) — committed to this repo, not derived at build time (this repo
# has no kg_build/ to derive them with). This was tried the other way first
# (mounted as a volume, nothing copied in) and it fails exactly the way
# you'd expect on a real hosting platform with no shared filesystem to the
# monorepo: `RuntimeError: search index missing at chunks.json` on every
# boot. The trade-off this accepts: the corpus only updates on a rebuild, not
# automatically. When the source PDFs change and `python -m kg_build` is
# re-run in the monorepo, `data/chunks.json` here needs a fresh copy and a
# redeploy — it does not update itself. For how rarely a statute amends,
# that's the right side of this trade-off; `docker-compose.yml`'s volume
# mount is still how local dev picks up a rebuild instantly, without needing
# a copy-and-redeploy cycle for every iteration.

FROM python:3.11-slim

# libgomp1 is a real runtime dependency of neo4j's pack/unpack, not build
# tooling — kept even though nothing here compiles from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer the dependency install below the source copy so a code edit doesn't
# invalidate the pip cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The package: this repo's own backend/ subfolder (backend/app.py etc, one
# level below this Dockerfile) — NOT the outer repo root, which holds only
# packaging files (this Dockerfile, requirements.txt, .env*). Copying it to
# an image path also named backend/ is what makes `uvicorn backend.app:app`
# below resolve correctly.
COPY backend/ backend/

# The corpus, baked in — see the module docstring above. Committed to this
# repo at data/chunks.json and data/vocab.yaml, not generated here.
COPY data/ data/

# logs/ is NOT copied in — it's write-only runtime state (audit.jsonl,
# appended per request), never a build input. Created empty; mounted as a
# volume for local dev (docker-compose.yml) so it survives a container
# restart there, but on most hosting platforms this is ephemeral by nature
# of the platform, same as it always was.
RUN mkdir -p logs

# Runs as a non-root user; logs/ is written by the app (audit.jsonl at 0600).
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# No secrets baked in — Neo4j credentials, DPDP_PROVIDER, ANTHROPIC_API_KEY,
# and the optional Langfuse keys are all supplied at `docker run` / compose
# time via -e or --env-file, never COPYed .env.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
