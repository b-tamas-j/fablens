"""Agent runtime: one ReAct agent over the FabLens tools, with replies traced by code."""

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import aclosing
from typing import Any

import pydantic_core
from a2a.types import TaskState
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
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
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph

from fablens import tools
from fablens.llm import chat_models
from fablens.models import (
    FINAL_STATES,
    AgentResponse,
    OnToolEvent,
    QueryTrace,
    Reply,
    Sources,
    ToolEvent,
    ToolFinished,
    ToolStarted,
)
from fablens.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

RESPONSE_TOOL = "agent_response"
# About twice the model calls of the most complex demo question.
MODEL_CALL_LIMIT = 12


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
    """Send a ToolError back to the model so it can correct the call; other errors propagate."""
    if not isinstance(exc, tools.ToolError):
        return None
    call = request.tool_call
    logger.warning("Tool %s failed: %s Args: %s", call["name"], exc, call["args"])
    return f"Error: {exc}"


@wrap_model_call
async def log_model_errors(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Log each failed model call, also when the fallback model then answers."""
    try:
        return await handler(request)
    except Exception as exc:
        logger.warning("Model call to %s failed: %r", type(request.model).__name__, exc)
        raise


def build_agent(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Create the agent; without a model, use the models configured in the environment."""
    middleware: list[AgentMiddleware] = [
        ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT),
        ToolErrorMiddleware(on_error=tool_error_message),
    ]
    if model is None:
        model, fallback = chat_models()
        if fallback is not None:
            middleware.append(ModelFallbackMiddleware(fallback))
    # Inside the fallback middleware, so the failure of each model is logged.
    middleware.append(log_model_errors)
    response_format = ToolStrategy(AgentResponse)
    # gpt-oss tends to call the tool in snake_case, and Groq rejects unknown tool names.
    response_format.schema_specs[0].name = RESPONSE_TOOL
    return create_agent(
        model,
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        response_format=response_format,
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


def tool_events(update: dict[str, Any], started: dict[str, str]) -> Iterator[ToolEvent]:
    """Tool calls and results in one graph update, without the structured response tool."""
    for node_update in update.values():
        if not isinstance(node_update, dict):
            continue
        for message in node_update.get("messages", []):
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    if call["name"] != RESPONSE_TOOL:
                        started[call["id"]] = call["name"]
                        yield ToolStarted(call_id=call["id"], name=call["name"], args=call["args"])
            elif isinstance(message, ToolMessage) and message.tool_call_id in started:
                yield ToolFinished(
                    call_id=message.tool_call_id,
                    name=started.pop(message.tool_call_id),
                    output=message.text,
                    error=message.status == "error",
                )


async def resume_point(agent: CompiledStateGraph, thread_id: str) -> RunnableConfig:
    """Continue the thread from its last finished turn, so a cancelled or failed turn is dropped."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if not (await agent.aget_state(config)).next:
        return config
    logger.info("Dropping an unfinished turn in thread %s", thread_id)
    async for snapshot in agent.aget_state_history(config):
        if not snapshot.next:
            return snapshot.config
    await agent.checkpointer.adelete_thread(thread_id)
    return config


def make_reply(state: dict[str, Any]) -> Reply:
    """Turn the final agent state into a reply; without a structured response the turn failed."""
    response = state.get("structured_response")
    if response is None:
        return Reply(
            state=TaskState.TASK_STATE_FAILED,
            answer="The agent did not finish its answer. Try again or ask a narrower question.",
        )
    return Reply(
        state=FINAL_STATES[response.state],
        answer=response.answer,
        sources=collect_sources(state["messages"]),
    )


async def aask(
    agent: CompiledStateGraph,
    question: str,
    thread_id: str,
    on_tool_event: OnToolEvent | None = None,
) -> Reply:
    """Answer one question in a chat session, reporting tool calls as they run; never raises."""
    logger.debug("Question in thread %s: %s", thread_id, question)
    start = time.perf_counter()
    messages: list[AnyMessage] = []
    started: dict[str, str] = {}
    state: dict[str, Any] = {}
    try:
        config = await resume_point(agent, thread_id)
        stream = agent.astream(
            {"messages": [HumanMessage(question)]}, config, stream_mode=["updates", "values"]
        )
        # Closing the stream on cancel stops the graph before the next turn starts.
        async with aclosing(stream):
            async for mode, data in stream:
                if mode == "values":
                    state = data
                    continue
                for event in tool_events(data, started):
                    if on_tool_event is not None:
                        await on_tool_event(event)
        messages = state["messages"]
        reply = make_reply(state)
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


def ask(agent: CompiledStateGraph, question: str, thread_id: str) -> Reply:
    """Synchronous aask without tool events, for scripts and tests."""
    return asyncio.run(aask(agent, question, thread_id))
