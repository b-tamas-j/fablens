import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from fakes import agent_with, call, respond
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.graph.state import CompiledStateGraph

from fablens.a2a_server import RPC_PATH, add_a2a_routes
from fablens.llm import ConfigError

HEADERS = {"A2A-Version": "1.0"}
MACHINE_COUNT = "SELECT count(*) AS machines FROM machines"


def a2a_client(agent_factory: Callable[[], CompiledStateGraph]) -> TestClient:
    app = FastAPI()
    add_a2a_routes(app, agent_factory, "http://testserver/")
    return TestClient(app)


def request(method: str, text: str, context_id: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "ROLE_USER",
        "messageId": str(uuid4()),
        "parts": [{"text": text}],
    }
    if context_id:
        message["contextId"] = context_id
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": {"message": message}}


def send(client: TestClient, text: str, context_id: str | None = None) -> dict[str, Any]:
    response = client.post(RPC_PATH, headers=HEADERS, json=request("SendMessage", text, context_id))
    response.raise_for_status()
    body = response.json()
    assert "error" not in body, body
    return body["result"]["task"]


def stream(client: TestClient, text: str) -> Iterator[dict[str, Any]]:
    payload = request("SendStreamingMessage", text)
    with client.stream("POST", RPC_PATH, headers=HEADERS, json=payload) as response:
        for line in response.iter_lines():
            if line.startswith("data:"):
                yield json.loads(line.removeprefix("data:"))["result"]


def status_text(task: dict[str, Any]) -> str:
    return task["status"]["message"]["parts"][0]["text"]


def test_agent_card_describes_the_endpoint_and_skill() -> None:
    with a2a_client(agent_with) as client:
        card = client.get("/.well-known/agent-card.json").json()

    assert card["name"] == "FabLens"
    assert card["supportedInterfaces"] == [
        {"url": "http://testserver/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ]
    assert card["capabilities"] == {"streaming": True}
    assert [skill["id"] for skill in card["skills"]] == ["equipment-data-qa"]


def test_answer_comes_with_sources_artifact(database: Path) -> None:
    agent = agent_with(
        call("run_sql", sql=MACHINE_COUNT),
        respond("completed", "There are 2 machines."),
    )

    with a2a_client(lambda: agent) as client:
        task = send(client, "How many machines are there?")

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert status_text(task) == "There are 2 machines."
    [artifact] = task["artifacts"]
    assert artifact["name"] == "sources"
    sources = artifact["parts"][0]["data"]
    assert sources["tables"] == ["machines"]
    assert sources["queries"] == [{"sql": MACHINE_COUNT, "row_count": 1, "truncated": False}]


def test_refusal_is_a_rejected_task() -> None:
    agent = agent_with(respond("rejected", "There is no cost data."))

    with a2a_client(lambda: agent) as client:
        task = send(client, "How much did the repairs cost?")

    assert task["status"]["state"] == "TASK_STATE_REJECTED"
    assert status_text(task) == "There is no cost data."


def test_context_continues_the_chat_thread() -> None:
    agent = agent_with(respond("completed", "Hello."), respond("completed", "Hello again."))

    with a2a_client(lambda: agent) as client:
        first = send(client, "Hi!")
        second = send(client, "Hi again!", first["contextId"])

    assert second["contextId"] == first["contextId"]
    assert second["id"] != first["id"]
    state = agent.get_state({"configurable": {"thread_id": first["contextId"]}})
    assert [m.text for m in state.values["messages"] if m.type == "human"] == ["Hi!", "Hi again!"]


def test_missing_config_fails_the_task() -> None:
    def no_model() -> CompiledStateGraph:
        raise ConfigError("LLM_MODEL is not set.")

    with a2a_client(no_model) as client:
        task = send(client, "Which machines are critical?")

    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert status_text(task) == "LLM_MODEL is not set."
    assert "artifacts" not in task


def test_stream_reports_tool_calls_before_the_answer(database: Path) -> None:
    agent = agent_with(
        call("run_sql", sql=MACHINE_COUNT),
        respond("completed", "There are 2 machines."),
    )

    with a2a_client(lambda: agent) as client:
        events = list(stream(client, "How many machines are there?"))

    updates = [event["statusUpdate"]["status"] for event in events if "statusUpdate" in event]
    progress = [status_text({"status": s}) for s in updates if "message" in s]
    assert progress == ["Calling run_sql", "There are 2 machines."]
    assert updates[-1]["state"] == "TASK_STATE_COMPLETED"
