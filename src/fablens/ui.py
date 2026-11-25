"""Chat UI: a thin Chainlit adapter over the agent core."""

import functools
import logging

import chainlit as cl
from a2a.types import TaskState
from chainlit.config import config
from chainlit.utils import utc_now
from langgraph.graph.state import CompiledStateGraph

from fablens.agent import aask, build_agent
from fablens.llm import ConfigError
from fablens.log_config import configure_logging
from fablens.models import Reply, Sources, ToolEvent, ToolStarted
from fablens.tools import glossary

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
# The escaped colon keeps remark-directive from taking the block before Chainlit's alert plugin.
REFUSAL_ALERT = "\\:::warning\nNot answered: the data or a definition is missing.\n:::"
# Chainlit shows a step as "Using <name>" while it runs and "Used <name>" when it is done.
PROGRESS_LABELS = {
    "get_business_context": "business definitions",
    "list_tables": "the data catalog",
    "run_sql": "a database query",
}


@functools.cache
def shared_agent() -> CompiledStateGraph:
    """The agent of all chat sessions, created at the first question."""
    return build_agent()


def progress_label(event: ToolStarted) -> str:
    """Plain-language name of what a tool call uses, shown while the agent works."""
    if event.name == "describe_table" and "table" in event.args:
        return f"the {event.args['table']} table"
    return PROGRESS_LABELS.get(event.name, "the data")


def counted(count: int, singular: str, plural: str) -> str:
    """Count with the matching noun, for example `1 table` or `2 tables`."""
    return f"{count} {singular if count == 1 else plural}"


def sources_summary(sources: Sources) -> str:
    """One-line summary of the sources, for example `1 definition, 2 tables and 1 query`."""
    parts = [
        counted(count, singular, plural)
        for count, singular, plural in [
            (len(sources.glossary_terms), "definition", "definitions"),
            (len(sources.tables), "table", "tables"),
            (len(sources.queries), "query", "queries"),
        ]
        if count
    ]
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def sources_markdown(sources: Sources) -> str:
    """Definitions, tables and each query with its row count, as Markdown."""
    parts = []
    if sources.glossary_terms:
        names = {term.id: term.name for term in glossary().terms}
        terms = ", ".join(names.get(term, term) for term in sources.glossary_terms)
        parts.append(f"**Definitions:** {terms}")
    if sources.tables:
        parts.append("**Tables:** " + ", ".join(sources.tables))
    for query in sources.queries:
        rows = counted(query.row_count, "row", "rows")
        if query.truncated:
            rows += ", truncated"
        parts.append(f"**Query** ({rows}):\n```sql\n{query.sql}\n```")
    return "\n\n".join(parts)


async def show_progress(event: ToolEvent, progress: cl.Step) -> None:
    """Show what the current tool call uses in the progress step."""
    if not isinstance(event, ToolStarted):
        return
    progress.name = progress_label(event)
    if progress.start is None:
        progress.start = utc_now()
        await progress.send()
    else:
        await progress.update()


async def show_reply(reply: Reply, progress: cl.Step) -> None:
    """Turn the progress step into the sources summary, then send the answer."""
    sources = reply.sources
    if progress.start is not None:
        if sources.glossary_terms or sources.tables or sources.queries:
            progress.name = sources_summary(sources)
            progress.output = sources_markdown(sources)
            progress.end = utc_now()
            await progress.update()
        else:
            await progress.remove()
    if reply.state == TaskState.TASK_STATE_FAILED:
        await cl.ErrorMessage(content=reply.answer).send()
        return
    content = reply.answer
    if reply.state == TaskState.TASK_STATE_REJECTED:
        content = f"{REFUSAL_ALERT}\n\n{content}"
    await cl.Message(content=content).send()


@cl.set_starters
async def starters(user: cl.User | None, language: str | None) -> list[cl.Starter]:
    """Offer the demo questions on the start screen."""
    return [cl.Starter(label=question, message=question) for question in STARTERS]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer a chat message, showing the agent's progress and sources."""
    try:
        agent = shared_agent()
    except ConfigError as exc:
        await cl.ErrorMessage(content=str(exc)).send()
        return
    # The avatar name keeps the FabLens avatar while the step name changes.
    progress = cl.Step(
        type="tool",
        parent_id=cl.context.current_step.id,
        metadata={"avatarName": config.ui.name},
        show_input=False,
    )
    reply = await aask(
        agent,
        message.content,
        cl.context.session.thread_id,
        functools.partial(show_progress, progress=progress),
    )
    await show_reply(reply, progress)


@cl.on_stop
async def on_stop() -> None:
    """Log that the user stopped the run."""
    logger.info("Run stopped by the user in thread %s", cl.context.session.thread_id)
