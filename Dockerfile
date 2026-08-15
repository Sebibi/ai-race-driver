# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.13-slim-trixie@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a

FROM ${PYTHON_IMAGE} AS common

ARG UV_VERSION=0.9.28

LABEL org.opencontainers.image.source="https://github.com/Sebibi/ai-race-driver" \
      org.opencontainers.image.description="JAX-native reinforcement-learning racing environment"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && groupadd --gid 1000 coder \
    && useradd --uid 1000 --gid coder --create-home --shell /bin/bash coder \
    && mkdir --parents /app /outputs \
    && chown coder:coder /app /outputs

WORKDIR /app

COPY --chown=coder:coder pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY --chown=coder:coder src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable \
    && chown -R coder:coder /opt/venv

USER coder

CMD ["bash"]

FROM common AS cpu

ARG VCS_BRANCH=unknown
ARG VCS_COMMIT_MESSAGE=unknown
ARG VCS_REVISION=unknown
LABEL org.opencontainers.image.revision="${VCS_REVISION}"
ENV AI_RACE_GIT_BRANCH="${VCS_BRANCH}" \
    AI_RACE_GIT_COMMIT_MESSAGE="${VCS_COMMIT_MESSAGE}" \
    AI_RACE_GIT_COMMIT_HASH="${VCS_REVISION}"

FROM common AS cuda13

USER root
RUN --mount=type=cache,target=/root/.cache/uv \
    JAX_VERSION="$(/opt/venv/bin/python -c 'from importlib.metadata import version; print(version("jax"))')" \
    && uv pip install --python /opt/venv/bin/python "jax[cuda13]==${JAX_VERSION}" \
    && /opt/venv/bin/python -c 'from importlib.metadata import version; assert version("jax") == version("jax-cuda13-plugin")' \
    && chown -R coder:coder /opt/venv
USER coder

ARG VCS_BRANCH=unknown
ARG VCS_COMMIT_MESSAGE=unknown
ARG VCS_REVISION=unknown
LABEL org.opencontainers.image.revision="${VCS_REVISION}"
ENV AI_RACE_GIT_BRANCH="${VCS_BRANCH}" \
    AI_RACE_GIT_COMMIT_MESSAGE="${VCS_COMMIT_MESSAGE}" \
    AI_RACE_GIT_COMMIT_HASH="${VCS_REVISION}"
