FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.25 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.14-slim

RUN useradd --system --create-home --home-dir /app fablens
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=fablens:fablens .chainlit/config.toml .chainlit/config.toml
COPY --chown=fablens:fablens chainlit.md ./
COPY public ./public

ENV PATH="/app/.venv/bin:$PATH"
USER fablens
EXPOSE 8000
CMD ["uvicorn", "fablens.server:app", "--host", "0.0.0.0", "--port", "8000"]
