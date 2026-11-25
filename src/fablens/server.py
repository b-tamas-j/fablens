"""ASGI entry point that serves the chat UI under /chat and the A2A interface.

The `chainlit run` CLI is not used: it applies nest_asyncio, which breaks asyncio task
detection on Python 3.14.
"""

import functools
import os
from pathlib import Path

from chainlit.utils import mount_chainlit
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from fablens.a2a_server import add_a2a_routes
from fablens.agent import build_agent
from fablens.log_config import configure_logging

load_dotenv(find_dotenv(usecwd=True))
configure_logging()

app = FastAPI(title="FabLens")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    """Redirect the root to the chat UI."""
    return RedirectResponse("/chat")


add_a2a_routes(
    app, functools.cache(build_agent), os.environ.get("PUBLIC_URL", "http://localhost:8000")
)
mount_chainlit(app=app, target=str(Path(__file__).with_name("ui.py")), path="/chat")
