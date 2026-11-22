"""Agent runtime: one ReAct agent over the FabLens tools, with replies traced by code."""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

import pydantic_core
from a2a.types import TaskState
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolErrorMiddleware,
    wrap_model_call,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph

from fablens import tools
from fablens.llm import chat_models
from fablens.models import FINAL_STATES, AgentResponse, QueryTrace, Reply, Sources
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


def tool_error_message(exc: Exception, request: ToolCallRequest) -> str | None:
    if not isinstance(exc, tools.ToolError):
        return None
    call = request.tool_call
    logger.warning("Tool %s failed: %s Args: %s", call["name"], exc, call["args"])
    return f"Error: {exc}"


@wrap_model_call
def log_model_errors(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    try:
        return handler(request)
    except Exception as exc:
        logger.warning("Model call to %s failed: %r", type(request.model).__name__, exc)
        raise


def build_agent(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Create the agent; without a model, use the models configured in the environment."""
    middleware: list[AgentMiddleware] = [ToolErrorMiddleware(on_error=tool_error_message)]
    if model is None:
        model, fallback = chat_models()
        if fallback is not None:
            middleware.append(ModelFallbackMiddleware(fallback))
    # Inside the fallback middleware, so the failure of each model is logged.
    middleware.append(log_model_errors)
    return create_agent(
        model,
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        response_format=ToolStrategy(AgentResponse),
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[(AgentResponse.__module__, AgentResponse.__name__)]
            )
        ),
        name="fablens",
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


def answering_model(messages: list[AnyMessage]) -> str | None:
    """Name of the model that wrote the last AI message, so a fallback answer is visible."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.response_metadata.get("model_name")
    return None


def ask(agent: CompiledStateGraph, question: str, thread_id: str) -> Reply:
    """Answer one question in a chat session; never raises."""
    logger.debug("Question in thread %s: %s", thread_id, question)
    start = time.perf_counter()
    messages: list[AnyMessage] = []
    try:
        state = agent.invoke(
            {"messages": [HumanMessage(question)]},
            {"configurable": {"thread_id": thread_id}},
        )
        messages = state["messages"]
        response = state.get("structured_response")
        if response is None:
            reply = Reply(
                state=TaskState.TASK_STATE_FAILED,
                answer="The agent did not finish its answer. Try again.",
            )
        else:
            reply = Reply(
                state=FINAL_STATES[response.state],
                answer=response.answer,
                sources=collect_sources(messages),
            )
    except Exception as exc:
        logger.exception("Agent run failed in thread %s", thread_id)
        reply = Reply(
            state=TaskState.TASK_STATE_FAILED,
            answer=f"The agent failed ({type(exc).__name__}). Try again.",
        )
    logger.info(
        "Reply in thread %s: state=%s model=%s queries=%d duration=%.1fs",
        thread_id,
        TaskState.Name(reply.state),
        answering_model(messages),
        len(reply.sources.queries),
        time.perf_counter() - start,
    )
    return reply
