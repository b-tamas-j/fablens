# FabLens

FabLens is a prototype Data Agent: users ask questions about equipment and service data in a chat, and the agent answers from the data with traceable sources (tables, SQL, glossary entries, row count). If the data or a definition is missing, it says so instead of guessing.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/). uv installs Python 3.14 (pinned in `.python-version`) if it is missing.

```shell
uv sync
cp .env.example .env
```

Fill in the API keys in `.env`. `LLM_MODEL` is the primary model, `LLM_FALLBACK_MODEL` is used when the primary fails. `.env` is git-ignored; never commit it.

## Development

CI runs the same checks on every pull request and on `main`:

```shell
uv run ruff check
uv run ruff format --check
uv run pytest
```
