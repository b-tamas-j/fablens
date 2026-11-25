"""A2A interface: a thin adapter that serves the agent core to other agents."""

from collections.abc import Callable
from importlib.metadata import version

from a2a.helpers import new_data_part, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from a2a.utils import TransportProtocol
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT
from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph

from fablens.agent import aask
from fablens.llm import ConfigError
from fablens.models import ToolEvent, ToolStarted

RPC_PATH = "/a2a"
SOURCES_ARTIFACT = "sources"


def agent_card(public_url: str) -> AgentCard:
    """Describe FabLens for other agents: the JSON-RPC endpoint, streaming and the Q&A skill."""
    return AgentCard(
        name="FabLens",
        description="Answers questions about equipment and service data with traceable sources. "
        "Refuses with a reason when the data or a definition is missing.",
        version=version("fablens"),
        supported_interfaces=[
            AgentInterface(
                url=public_url.rstrip("/") + RPC_PATH,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="equipment-data-qa",
                name="Equipment data Q&A",
                description="Answers questions about machines, telemetry, errors, maintenance "
                "and failures from the data, with KPI definitions from a glossary. "
                f"The answer is the final status message; the {SOURCES_ARTIFACT} artifact lists "
                "the tables, the executed SQL with row counts and the glossary entries used.",
                tags=["predictive maintenance", "equipment data", "KPI", "SQL"],
                examples=[
                    "Which machine model had the most failures in 2015?",
                    "What is the MTBF of model3?",
                    "How much did the repairs of machine 17 cost?",
                ],
            )
        ],
    )


class FabLensExecutor(AgentExecutor):
    """Runs one A2A task per question; the A2A context is the chat thread."""

    def __init__(self, agent_factory: Callable[[], CompiledStateGraph]) -> None:
        self.agent_factory = agent_factory

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Answer the question, report tool calls as working updates, end with the reply state."""
        if context.current_task is None:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()
        try:
            agent = self.agent_factory()
        except ConfigError as exc:
            await updater.failed(updater.new_agent_message([new_text_part(str(exc))]))
            return

        async def report(event: ToolEvent) -> None:
            if isinstance(event, ToolStarted):
                message = updater.new_agent_message([new_text_part(f"Calling {event.name}")])
                await updater.start_work(message)

        reply = await aask(agent, context.get_user_input(), context.context_id, report)
        if reply.state != TaskState.TASK_STATE_FAILED:
            await updater.add_artifact(
                [new_data_part(reply.sources.model_dump())], name=SOURCES_ARTIFACT
            )
        answer = updater.new_agent_message([new_text_part(reply.answer)])
        await updater.update_status(reply.state, answer)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Mark the task as cancelled."""
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def add_a2a_routes(
    app: FastAPI, agent_factory: Callable[[], CompiledStateGraph], public_url: str
) -> None:
    """Serve the agent card at the well-known path and the JSON-RPC endpoint at RPC_PATH."""
    card = agent_card(public_url)
    handler = DefaultRequestHandler(
        agent_executor=FabLensExecutor(agent_factory),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=RPC_PATH),
    )
