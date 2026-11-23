import asyncio
import logging
from collections.abc import Sequence
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from a2a.types import TaskState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fablens import agent as agent_module
from fablens.agent import aask, ask, build_agent
from fablens.models import Reply, Sources, ToolEvent, ToolFinished, ToolStarted

FAILURES_PER_MODEL = (
    "SELECT m.model, count(*) AS failures FROM failures f "
    "JOIN machines m USING (machineID) GROUP BY m.model"
)
call_ids = count()


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeModel:
        return self


class FailingModel(FakeModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")


def call(name: str, **args: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call{next(call_ids)}"}])


def respond(state: str, answer: str) -> AIMessage:
    return call("agent_response", state=state, answer=answer)


def agent_with(*messages: AIMessage):
    return build_agent(FakeModel(messages=iter(messages)))


async def stop(event: ToolEvent) -> None:
    raise asyncio.CancelledError


def questions(agent: Any, thread_id: str) -> list[str]:
    messages = agent.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]
    return [message.text for message in messages if isinstance(message, HumanMessage)]


def test_reply_traces_sql_and_glossary(database: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    question = "Which model had the most failures?"
    agent = agent_with(
        call("get_business_context", question=question),
        call("run_sql", sql=FAILURES_PER_MODEL),
        respond("completed", "model3 had the most failures."),
    )

    reply = ask(agent, question, "t1")

    assert reply.state == TaskState.TASK_STATE_COMPLETED
    assert reply.answer == "model3 had the most failures."
    assert reply.sources.tables == ["failures", "machines"]
    assert [(q.sql, q.row_count, q.truncated) for q in reply.sources.queries] == [
        (FAILURES_PER_MODEL, 1, False)
    ]
    assert "failure" in reply.sources.glossary_terms
    assert "Reply in thread t1: state=TASK_STATE_COMPLETED" in caplog.text
    assert "queries=1" in caplog.text


def test_tool_error_goes_back_to_the_model(
    database: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = agent_with(
        call("run_sql", sql="DELETE FROM machines"),
        respond("rejected", "I can only read data."),
    )

    reply = ask(agent, "Delete all machines.", "t1")

    assert reply.state == TaskState.TASK_STATE_REJECTED
    assert reply.sources.queries == []
    messages = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    error = next(m for m in messages if isinstance(m, ToolMessage) and m.name == "run_sql")
    assert error.status == "error"
    assert "Only a single SELECT statement is allowed." in error.content
    assert "Tool run_sql failed" in caplog.text


def test_sources_cover_only_the_last_question(database: Path) -> None:
    agent = agent_with(
        call("run_sql", sql="SELECT count(*) FROM machines"),
        respond("completed", "There are 2 machines."),
        respond("completed", "You're welcome."),
    )

    first = ask(agent, "How many machines are there?", "t1")
    second = ask(agent, "Thanks!", "t1")

    assert first.sources.tables == ["machines"]
    assert second.sources == Sources()


def test_tool_events_report_each_call_and_result(database: Path) -> None:
    agent = agent_with(
        call("run_sql", sql="DELETE FROM machines"),
        call("run_sql", sql="SELECT count(*) FROM machines"),
        respond("completed", "There are 2 machines."),
    )
    events: list[ToolEvent] = []

    async def record(event: ToolEvent) -> None:
        events.append(event)

    reply = asyncio.run(aask(agent, "How many machines are there?", "t1", record))

    assert reply.state == TaskState.TASK_STATE_COMPLETED
    assert [(type(event), event.name) for event in events] == [
        (ToolStarted, "run_sql"),
        (ToolFinished, "run_sql"),
        (ToolStarted, "run_sql"),
        (ToolFinished, "run_sql"),
    ]
    assert events[0].args == {"sql": "DELETE FROM machines"}
    assert events[0].call_id == events[1].call_id
    assert events[1].error
    assert not events[3].error
    assert '"row_count":1' in events[3].output


def test_cancelled_turn_is_dropped(database: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    agent = agent_with(
        respond("completed", "Hello."),
        call("run_sql", sql="SELECT count(*) FROM machines"),
        respond("completed", "Hello again."),
    )

    async def chat() -> Reply:
        await aask(agent, "Hi!", "t1")
        with pytest.raises(asyncio.CancelledError):
            await aask(agent, "How many machines are there?", "t1", stop)
        return await aask(agent, "Hi again!", "t1")

    reply = asyncio.run(chat())

    assert reply.answer == "Hello again."
    assert questions(agent, "t1") == ["Hi!", "Hi again!"]
    assert "Dropping an unfinished turn in thread t1" in caplog.text


def test_cancelled_first_turn_starts_a_fresh_thread(database: Path) -> None:
    agent = agent_with(
        call("run_sql", sql="SELECT count(*) FROM machines"),
        respond("completed", "Hello."),
    )

    async def chat() -> Reply:
        with pytest.raises(asyncio.CancelledError):
            await aask(agent, "How many machines are there?", "t1", stop)
        return await aask(agent, "Hi!", "t1")

    reply = asyncio.run(chat())

    assert reply.answer == "Hello."
    assert questions(agent, "t1") == ["Hi!"]


def test_model_failure_gives_an_error_reply(database: Path) -> None:
    agent = build_agent(FailingModel(messages=iter([])))

    reply = ask(agent, "Which machines are critical?", "t1")

    assert reply.state == TaskState.TASK_STATE_FAILED
    assert "RuntimeError" in reply.answer


def test_fallback_model_answers_and_primary_failure_is_logged(
    database: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    primary = FailingModel(messages=iter([]))
    fallback = FakeModel(messages=iter([respond("completed", "Answer from the fallback.")]))
    monkeypatch.setattr(agent_module, "chat_models", lambda: (primary, fallback))

    reply = ask(build_agent(), "Which machines are critical?", "t1")

    assert reply.state == TaskState.TASK_STATE_COMPLETED
    assert reply.answer == "Answer from the fallback."
    assert "Model call to FailingModel failed" in caplog.text
