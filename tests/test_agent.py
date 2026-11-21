from collections.abc import Sequence
from itertools import count
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from fablens.agent import Sources, ask, build_agent

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


def respond(status: str, answer: str) -> AIMessage:
    return call("AgentResponse", status=status, answer=answer)


def agent_with(*messages: AIMessage):
    return build_agent(FakeModel(messages=iter(messages)))


def test_reply_traces_sql_and_glossary(database: Path) -> None:
    question = "Which model had the most failures?"
    agent = agent_with(
        call("get_business_context", question=question),
        call("run_sql", sql=FAILURES_PER_MODEL),
        respond("answered", "model3 had the most failures."),
    )

    reply = ask(agent, question, "t1")

    assert reply.status == "answered"
    assert reply.answer == "model3 had the most failures."
    assert reply.sources.tables == ["failures", "machines"]
    assert [(q.sql, q.row_count, q.truncated) for q in reply.sources.queries] == [
        (FAILURES_PER_MODEL, 1, False)
    ]
    assert "failure" in reply.sources.glossary_terms


def test_tool_error_goes_back_to_the_model(database: Path) -> None:
    agent = agent_with(
        call("run_sql", sql="DELETE FROM machines"),
        respond("cannot_answer", "I can only read data."),
    )

    reply = ask(agent, "Delete all machines.", "t1")

    assert reply.status == "cannot_answer"
    assert reply.sources.queries == []
    messages = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    error = next(m for m in messages if isinstance(m, ToolMessage) and m.name == "run_sql")
    assert error.status == "error"
    assert "Only a single SELECT statement is allowed." in error.content


def test_sources_cover_only_the_last_question(database: Path) -> None:
    agent = agent_with(
        call("run_sql", sql="SELECT count(*) FROM machines"),
        respond("answered", "There are 2 machines."),
        respond("answered", "You're welcome."),
    )

    first = ask(agent, "How many machines are there?", "t1")
    second = ask(agent, "Thanks!", "t1")

    assert first.sources.tables == ["machines"]
    assert second.sources == Sources()


def test_model_failure_gives_an_error_reply(database: Path) -> None:
    agent = build_agent(FailingModel(messages=iter([])))

    reply = ask(agent, "Which machines are critical?", "t1")

    assert reply.status == "error"
    assert "RuntimeError" in reply.answer
