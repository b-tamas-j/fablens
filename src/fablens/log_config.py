"""Logging setup for the FabLens entry points."""

import logging
import os


def configure_logging() -> None:
    """Log with ISO timestamps at the level in LOG_LEVEL (default INFO)."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
