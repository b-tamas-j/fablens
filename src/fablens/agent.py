"""Agent runtime: one ReAct agent over the FabLens tools, with replies traced by code."""

import functools
import logging
from collections.abc import Callable
from typing import Any, Literal

import pydantic_core
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ToolErrorMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from fablens import tools
from fablens.llm import chat_models
from fablens.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def as_langchain_tool(func: Callable[..., Any]) -> BaseTool:
    """Expose a tool function to the LLM as JSON, keeping the result as artifact for tracing."""

    @functools.wraps(func)
    def run(*args: Any, **kwargs: Any) -> tuple[str, Any]:
        result = func(*args, **kwargs)
        return pydantic_core.to_json(result).decode(), result

    return tool(run, response_format="content_and_artifact", parse_docstring=True)


TOOLS = [
    as_langchain_tool(func)
    for func in (tools.get_business_context, tools.list_tables, tools.describe_table, tools.run_sql)
]


class AgentResponse(BaseModel):
    """Final answer to the user's question."""

    status: Literal["answered", "cannot_answer"] = Field(
        description="answered: the answer comes from tool results. "
        "cannot_answer: the data or a definition needed for a reliable answer is missing."
    )
    answer: str = Field(
        description="Short answer for the user. For cannot_answer: why, and what is available."
    )


class QueryTrace(BaseModel):
    sql: str
    row_count: int
    truncated: bool


class Sources(BaseModel):
    tables: list[str] = []
    queries: list[QueryTrace] = []
    glossary_terms: list[str] = []


class Reply(BaseModel):
    status: Literal["answered", "cannot_answer", "error"]
    answer: str
    sources: Sources = Field(default_factory=Sources)


def tool_error_message(exc: Exception, _request: object) -> str | None:
    return f"Error: {exc}" if isinstance(exc, tools.ToolError) else None


def build_agent(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Create the agent; without a model, use the models configured in the environment."""
    middleware: list[AgentMiddleware] = [ToolErrorMiddleware(on_error=tool_error_message)]
    if model is None:
        model, fallback = chat_models()
        if fallback is not None:
            middleware.append(ModelFallbackMiddleware(fallback))
    return create_agent(
        model,
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        response_format=ToolStrategy(AgentResponse),
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=[(__name__, "AgentResponse")])
        ),
    )


def collect_sources(messages: list[AnyMessage]) -> Sources:
    """Collect the sources of the last turn from the successful tool results."""
    last_question = max(i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage))
    sources = Sources()
    for message in messages[last_question + 1 :]:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        artifact = message.artifact
        if isinstance(artifact, tools.QueryResult):
            sources.queries.append(QueryTrace.model_validate(artifact, from_attributes=True))
            sources.tables += [name for name in artifact.tables if name not in sources.tables]
        elif isinstance(artifact, tools.BusinessContext):
            sources.glossary_terms += [
                term.id for term in artifact.terms if term.id not in sources.glossary_terms
            ]
    return sources


def ask(agent: CompiledStateGraph, question: str, thread_id: str) -> Reply:
    """Answer one question in a chat session; never raises."""
    try:
        state = agent.invoke(
            {"messages": [HumanMessage(question)]},
            {"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        logger.exception("Agent run failed")
        return Reply(status="error", answer=f"The agent failed ({type(exc).__name__}). Try again.")
    response = state.get("structured_response")
    if response is None:
        return Reply(status="error", answer="The agent did not finish its answer. Try again.")
    return Reply(
        status=response.status,
        answer=response.answer,
        sources=collect_sources(state["messages"]),
    )
