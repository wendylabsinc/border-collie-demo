from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .mission import MissionMachine, RestartRequired


def create_app(
    mission: MissionMachine | None = None,
    web_root: Path | None = None,
) -> FastAPI:
    machine = mission or MissionMachine()
    root = web_root or Path(
        os.environ.get("BORDER_COLLIE_WEB_ROOT", "web")
    ).resolve()
    app = FastAPI(title="Border Collie Demo", version="0.1.0")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(root / "index.html")

    @app.get("/debug")
    async def debug() -> FileResponse:
        return FileResponse(root / "debug.html")

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return machine.status()

    @app.post("/api/stop")
    async def stop() -> dict[str, object]:
        try:
            machine.stop()
        except RestartRequired as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        return machine.status()

    return app
