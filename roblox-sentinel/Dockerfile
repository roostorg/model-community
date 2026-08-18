# Sentinel container image (default / GPU-capable variant).
#
# Builds a self-contained image that installs the `sentinel` library (with the
# optional `sbert` extra, which pulls in sentence-transformers + torch) and, by
# default, runs the beginner demo so you can see the full scoring flow.
#
# This variant installs the default torch build, which on Linux is GPU (CUDA)
# enabled and therefore large (~6-8 GB total image). Sentinel itself runs on
# CPU; if you don't need GPU, use `Dockerfile.cpu` for a much smaller image:
#   docker build -f Dockerfile.cpu -t sentinel:cpu .
FROM python:3.11-slim

# - PYTHONUNBUFFERED: print logs immediately instead of buffering them.
# - PYTHONDONTWRITEBYTECODE: don't litter the image with .pyc files.
# - POETRY_VIRTUALENVS_CREATE=false: install deps into the system Python so we
#   don't nest a virtualenv inside the container.
# - HF_HOME: where Hugging Face / sentence-transformers cache downloaded models.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    HF_HOME=/home/sentinel/.cache/huggingface

# build-essential is needed to compile some scientific Python wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# The project's build backend requires poetry-core >= 2.0, so use Poetry 2.x.
RUN pip install "poetry>=2.0,<3.0"

WORKDIR /app

# Copy only what's needed to install dependencies first. Keeping this separate
# from the rest of the source lets Docker cache the (slow) dependency install
# and skip it when only application code changes.
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

# Install the library plus the `sbert` extra. Skip the dev/docs/examples groups
# to keep the image lean.
RUN poetry install --extras sbert --without dev,docs,examples

# Copy the remaining source (examples, tests, docs, etc.).
COPY . .

# Run as a non-root user for safety.
RUN useradd --create-home sentinel \
    && mkdir -p "$HF_HOME" \
    && chown -R sentinel:sentinel /app /home/sentinel
USER sentinel

# By default, run the beginner demo end-to-end. Override this at `docker run`
# time to run your own script, e.g.:
#   docker run --rm sentinel python examples/Example_Threshold_Script.py
CMD ["python", "examples/beginner_demo.py"]
