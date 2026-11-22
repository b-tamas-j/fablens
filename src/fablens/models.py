"""Agent contract: the LLM's structured answer and the traced reply for the interfaces."""

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

from a2a.types import TaskState
from pydantic import BaseModel, Field, PlainSerializer


class AgentResponse(BaseModel):
    """Final answer to the user's question."""

    state: Literal["completed", "rejected"] = Field(
        description="completed: the answer comes from tool results. "
        "rejected: the data or a definition needed for a reliable answer is missing."
    )
    answer: str = Field(
        description="Short answer for the user. If rejected: why, and what is available."
    )


FINAL_STATES = {
    "completed": TaskState.TASK_STATE_COMPLETED,
    "rejected": TaskState.TASK_STATE_REJECTED,
}

A2ATaskState = Annotated[TaskState.ValueType, PlainSerializer(TaskState.Name, return_type=str)]


class QueryTrace(BaseModel):
    sql: str
    row_count: int
    truncated: bool


class Sources(BaseModel):
    tables: list[str] = []
    queries: list[QueryTrace] = []
    glossary_terms: list[str] = []


class Reply(BaseModel):
    state: A2ATaskState
    answer: str
    sources: Sources = Field(default_factory=Sources)


class ToolStarted(BaseModel):
    call_id: str
    name: str
    args: dict[str, Any]


class ToolFinished(BaseModel):
    call_id: str
    name: str
    output: str
    error: bool


type ToolEvent = ToolStarted | ToolFinished
type OnToolEvent = Callable[[ToolEvent], Awaitable[None]]
