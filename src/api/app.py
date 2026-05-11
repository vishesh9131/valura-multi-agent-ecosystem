"""FastAPI app factory."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from ..config import get_settings
from .routes import router


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO if settings.app_env != "test" else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    app = FastAPI(
        title="Valura AI",
        version="0.1.0",
        description="AI co-investor microservice — safety, intent, routing, streaming.",
    )
    app.include_router(router)
    return app


app = create_app()
