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
# Getting data/chunks.json onto this container at runtime is this repo's
# responsibility once deployed on its own — docker-compose.yml's volume
# mount only works for local development from within the monorepo checkout,
# where data/ is still a real sibling directory. A real hosting platform
# needs its own answer (a persistent volume, an init step that pulls
# chunks.json from object storage, etc.) — there is no single right answer
# baked in here on purpose, since it depends entirely on where this repo
# ends up deployed.

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

# data/ and logs/ are not copied in: data/chunks.json and data/vocab.yaml
# are runtime state (see the module docstring above for how they get here),
# and logs/ is written by the app. Mounted as volumes for local dev — see
# docker-compose.yml.
RUN mkdir -p data logs

# Runs as a non-root user; logs/ is written by the app (audit.jsonl at 0600).
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# No secrets baked in — Neo4j credentials, DPDP_PROVIDER, ANTHROPIC_API_KEY,
# and the optional Langfuse keys are all supplied at `docker run` / compose
# time via -e or --env-file, never COPYed .env.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
