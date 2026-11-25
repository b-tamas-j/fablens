"""Scripted chat models, so agent tests never call a real LLM."""

from collections.abc import Sequence
from itertools import count
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from fablens.agent import build_agent

call_ids = count()


class FakeModel(GenericFakeChatModel):
    """Chat model that returns the scripted messages in order."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeModel:
        return self


def call(name: str, **args: str) -> AIMessage:
    """AI message that calls one tool."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call{next(call_ids)}"}])


def respond(state: str, answer: str) -> AIMessage:
    """AI message with the final structured answer."""
    return call("agent_response", state=state, answer=answer)


def agent_with(*messages: AIMessage) -> CompiledStateGraph:
    """Agent whose model returns the given messages in order."""
    return build_agent(FakeModel(messages=iter(messages)))
