"""ASGI entry point that serves the chat UI under /chat.

The `chainlit run` CLI is not used: it applies nest_asyncio, which breaks asyncio task
detection on Python 3.14.
"""

from pathlib import Path

from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI(title="FabLens")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse("/chat")


mount_chainlit(app=app, target=str(Path(__file__).with_name("ui.py")), path="/chat")
