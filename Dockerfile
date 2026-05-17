# Multi-stage build for the hourly MPC optimization container.
#
# Stage 1 (builder): install dependencies into a virtualenv with uv, using the
# committed uv.lock for a reproducible build. The project is installed
# non-editable so the runtime stage doesn't need src/ on disk.
#
# Stage 2 (runtime): copy the venv into a slim image, drop privileges, run the
# daemon. The same image also serves the init-state oneshot — compose just
# overrides the command.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.11.10

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install third-party deps first (cached layer; only invalidated when
# pyproject.toml or uv.lock change). --extra optimization brings the solver
# stack (pyomo, highspy) and apscheduler; forecasting-only deps stay out.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra optimization --no-install-project --no-editable

# Install the project itself.
COPY src/optimization ./src/optimization
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra optimization --no-editable


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    TZ=Etc/UTC

# Non-root user; uid 1000 matches the typical default Linux user so bind-mount
# permissions line up without manual chowning.
RUN groupadd --system --gid 1000 mpc \
    && useradd --system --uid 1000 --gid mpc --home-dir /app --shell /usr/sbin/nologin mpc \
    && mkdir -p /shared/forecast /shared/state /shared/dispatch \
    && chown -R mpc:mpc /shared

COPY --from=builder --chown=mpc:mpc /app/.venv /app/.venv

USER mpc
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["-m", "optimization.daemon"]
