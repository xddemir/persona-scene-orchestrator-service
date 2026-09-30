"""FastAPI app.

    uvicorn scene_orchestrator.api.app:app --reload --reload-dir src --host 127.0.0.1 --port 8001

Port 8001 because image-gen's dev server already sits on 8000, and the
orchestrator calls it, so both run side by side on one machine.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .. import __version__
from .models import HealthResponse


def create_app() -> FastAPI:
    app = FastAPI(
        version=__version__,

    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    return app


# Module-level app for `uvicorn scene_orchestrator.api.app:app`.
app = create_app()
