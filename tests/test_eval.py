"""Eval of the demo questions against the real LLM and dataset: `uv run pytest -m eval`."""

import math
import re
from dataclasses import dataclass

import pytest
from a2a.types import TaskState
from langgraph.graph.state import CompiledStateGraph

from fablens.agent import ask, build_agent
from fablens.ingest import db_path
from fablens.llm import ConfigError

pytestmark = pytest.mark.eval

COMPLETED = TaskState.TASK_STATE_COMPLETED
REJECTED = TaskState.TASK_STATE_REJECTED


@dataclass(frozen=True)
class Case:
    question: str
    state: TaskState.ValueType
    glossary_term: str
    facts: tuple[str | float, ...] = ()


CASES = [
    pytest.param(
        Case(
            "Which machine model had the most failures in 2015?", COMPLETED, "failure", ("model3",)
        ),
        id="most-failures",
    ),
    pytest.param(
        Case(
            "Which machines are critical?",
            COMPLETED,
            "critical_machine",
            (13, 16, 17, 20, 22, 23, 37, 49, 71, 73, 83, 90, 98, 99),
        ),
        id="critical-machines",
    ),
    pytest.param(Case("What is the MTBF of model3?", COMPLETED, "mtbf", (1387.33,)), id="mtbf"),
    pytest.param(
        Case("How much did the repairs of machine 17 cost?", REJECTED, "repair_cost"),
        id="repair-cost",
    ),
    pytest.param(
        Case("Will machine 42 fail next week?", REJECTED, "failure_prediction"),
        id="prediction",
    ),
]


def numbers(text: str) -> list[float]:
    """Numbers in an answer; thousands separators such as in 1,387 are removed."""
    text = re.sub(r"(?<=\d)[,\u00a0\u202f](?=\d{3}(?!\d))", "", text)
    return [float(number) for number in re.findall(r"\d+(?:\.\d+)?", text)]


def mentions(answer: str, fact: str | float) -> bool:
    if isinstance(fact, str):
        return fact.lower() in "".join(answer.split()).lower()
    return any(math.isclose(number, fact, rel_tol=0.005) for number in numbers(answer))


@pytest.fixture(scope="module")
def agent() -> CompiledStateGraph:
    if not db_path().exists():
        pytest.fail(
            f"{db_path()} is missing: run `uv run python -m fablens.ingest`.", pytrace=False
        )
    try:
        return build_agent()
    except ConfigError as exc:
        pytest.fail(str(exc), pytrace=False)


@pytest.mark.parametrize("case", CASES)
def test_demo_question(agent: CompiledStateGraph, case: Case) -> None:
    reply = ask(agent, case.question, thread_id=case.question)
    report = reply.model_dump_json(indent=2)

    assert reply.state == case.state, report
    assert case.glossary_term in reply.sources.glossary_terms, report
    if case.state == COMPLETED:
        assert reply.sources.queries, report
    missing = [fact for fact in case.facts if not mentions(reply.answer, fact)]
    assert not missing, f"Missing in the answer: {missing}\n{report}"
