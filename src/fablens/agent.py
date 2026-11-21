"""Agent runtime: one ReAct agent over the FabLens tools, with replies traced by code."""

import json
import logging
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ToolErrorMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from fablens import tools
from fablens.llm import chat_models
from fablens.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact", parse_docstring=True)
def get_business_context(question: str) -> tuple[str, tools.BusinessContext]:
    """Look up business terms, KPI definitions and known data gaps in the glossary.

    Args:
        question: The user's question, or the business terms in it.
    """
    result = tools.get_business_context(question)
    return result.model_dump_json(exclude={"query"}), result


@tool
def list_tables() -> str:
    """List the database tables with their business description and row count."""
    return json.dumps([table.model_dump() for table in tools.list_tables()])


@tool(parse_docstring=True)
def describe_table(table: str) -> str:
    """Describe the columns of one table: name, SQL type and business meaning.

    Args:
        table: A table name returned by list_tables.
    """
    return tools.describe_table(table).model_dump_json()


@tool(response_format="content_and_artifact", parse_docstring=True)
def run_sql(sql: str) -> tuple[str, tools.QueryResult]:
    """Run one read-only DuckDB SELECT statement and return at most 100 rows.

    Args:
        sql: A single SELECT statement.
    """
    result = tools.run_sql(sql)
    return result.model_dump_json(include={"columns", "rows", "row_count", "truncated"}), result


TOOLS = [get_business_context, list_tables, describe_table, run_sql]


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
