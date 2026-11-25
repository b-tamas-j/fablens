"""LLM access: provider-agnostic chat models configured through environment variables."""

import os

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


class ConfigError(Exception):
    """The LLM configuration is missing or invalid."""


def create_model(name: str) -> BaseChatModel:
    """Create a chat model from a LangChain `provider:model` name; raise ConfigError if it fails."""
    try:
        return init_chat_model(name)
    except Exception as exc:
        raise ConfigError(f"Cannot create model {name!r}: {exc}") from None


def chat_models() -> tuple[BaseChatModel, BaseChatModel | None]:
    """Return the primary model (LLM_MODEL) and the optional fallback (LLM_FALLBACK_MODEL)."""
    load_dotenv(find_dotenv(usecwd=True))
    primary = os.environ.get("LLM_MODEL")
    if not primary:
        raise ConfigError("LLM_MODEL is not set. Copy .env.example to .env and fill it in.")
    fallback = os.environ.get("LLM_FALLBACK_MODEL")
    return create_model(primary), create_model(fallback) if fallback else None
