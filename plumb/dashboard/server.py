from __future__ import annotations

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

_STATIC = Path(__file__).parent / "static"


def serve(
    report_path: str | None = None,
    snapshot_path: str | None = None,
    port: int = 8080,
) -> None:
    app = _build_app(report_path=report_path, snapshot_path=snapshot_path)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def _build_app(
    report_path: str | None,
    snapshot_path: str | None,
) -> FastAPI:
    app = FastAPI(title="plumb dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (_STATIC / "index.html").read_text()

    @app.get("/api/report")
    async def report() -> dict:
        # Live session snapshot takes priority
        src = snapshot_path or report_path
        if src is None:
            return {"error": "no data source configured"}
        try:
            return json.loads(Path(src).read_text())
        except FileNotFoundError:
            return {"error": f"file not found: {src}"}
        except json.JSONDecodeError as e:
            return {"error": f"invalid JSON: {e}"}

    return app
