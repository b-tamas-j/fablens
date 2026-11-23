"""Chat UI: a thin Chainlit adapter over the agent core."""

import functools
import json
import logging

import chainlit as cl
from a2a.types import TaskState
from chainlit.utils import utc_now
from langgraph.graph.state import CompiledStateGraph

from fablens.agent import aask, build_agent
from fablens.llm import ConfigError
from fablens.log_config import configure_logging
from fablens.models import Reply, ToolEvent, ToolStarted

configure_logging()
# Chainlit loads this file by path, so __name__ is the file path.
logger = logging.getLogger("fablens.ui")

STARTERS = [
    "Which machine model had the most failures in 2015?",
    "Which machines are critical?",
    "What is the MTBF of model3?",
    "How much did the repairs of machine 17 cost?",
    "Will machine 42 fail next week?",
]
REFUSAL_ALERT = ":::warning\nNot answered: the data or a definition is missing.\n:::"


@functools.cache
def shared_agent() -> CompiledStateGraph:
    return build_agent()


def sources_markdown(reply: Reply) -> str:
    sources = reply.sources
    parts = [f"**A2A task state:** `{TaskState.Name(reply.state)}`"]
    if sources.tables:
        parts.append("**Tables:** " + ", ".join(f"`{name}`" for name in sources.tables))
    if sources.glossary_terms:
        terms = ", ".join(f"`{term}`" for term in sources.glossary_terms)
        parts.append(f"**Glossary entries:** {terms}")
    for query in sources.queries:
        rows = f"{query.row_count} row{'' if query.row_count == 1 else 's'}"
        if query.truncated:
            rows += " (truncated)"
        parts.append(f"**Executed SQL**, {rows}:\n```sql\n{query.sql}\n```")
    return "\n\n".join(parts)


async def show_tool_event(event: ToolEvent, steps: dict[str, cl.Step]) -> None:
    if isinstance(event, ToolStarted):
        is_sql = event.name == "run_sql"
        step = cl.Step(
            name=event.name,
            type="tool",
            parent_id=cl.context.current_step.id,
            show_input="sql" if is_sql else "json",
        )
        step.input = event.args["sql"] if is_sql else event.args
        step.start = utc_now()
        steps[event.call_id] = step
        await step.send()
        return
    step = steps.pop(event.call_id)
    step.output = event.output if event.error else json.loads(event.output)
    step.is_error = event.error
    step.end = utc_now()
    await step.update()


async def show_reply(reply: Reply) -> None:
    if reply.state == TaskState.TASK_STATE_FAILED:
        await cl.ErrorMessage(content=reply.answer).send()
        return
    content = reply.answer
    if reply.state == TaskState.TASK_STATE_REJECTED:
        content = f"{REFUSAL_ALERT}\n\n{content}"
    answer = cl.Message(content=content)
    await answer.send()
    sources = cl.Step(name="sources", type="tool", parent_id=answer.id, show_input=False)
    sources.output = sources_markdown(reply)
    sources.start = sources.end = utc_now()
    await sources.send()


@cl.set_starters
async def starters(user: cl.User | None, language: str | None) -> list[cl.Starter]:
    return [cl.Starter(label=question, message=question) for question in STARTERS]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        agent = shared_agent()
    except ConfigError as exc:
        await cl.ErrorMessage(content=str(exc)).send()
        return
    steps: dict[str, cl.Step] = {}
    reply = await aask(
        agent,
        message.content,
        cl.context.session.thread_id,
        functools.partial(show_tool_event, steps=steps),
    )
    await show_reply(reply)


@cl.on_stop
async def on_stop() -> None:
    logger.info("Run stopped by the user in thread %s", cl.context.session.thread_id)
