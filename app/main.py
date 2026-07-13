from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routes.tracking import router as tracking_router
from app.services import model_cache, runtime
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Vision RAG",
    version="1.0.0",
    description="YOLO + ReID + CLIP + Video RAG system",
)

origin = os.getenv("CORS_ORIGIN")
allowed_origins = (
    [value.strip() for value in origin.split(",") if value.strip()]
    if origin
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIRS = [
    "data/output",
    "data/crops",
    "data/evidence",
    "data/clips",
    "data/embeddings",
]

for directory in DATA_DIRS:
    Path(directory).mkdir(parents=True, exist_ok=True)

STATIC_MOUNTS = {
    "/outputs": "data/output",
    "/crops": "data/crops",
    "/evidence": "data/evidence",
    "/clips": "data/clips",
}

for route, directory in STATIC_MOUNTS.items():
    app.mount(route, StaticFiles(directory=directory), name=route.strip("/"))

app.include_router(tracking_router)


@app.get("/health")
def health_check():
    data_dirs = {
        directory: Path(directory).exists()
        for directory in DATA_DIRS
    }
    return {
        "status": "ok" if all(data_dirs.values()) else "degraded",
        "jobs": runtime.job_manager.summary(),
        "model_cache": model_cache.status(),
        "data_dirs": data_dirs,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    job_summary = runtime.job_manager.summary()
    cache_status = model_cache.status()
    lines = [
        "# HELP mot_reid_jobs_total Jobs tracked by status.",
        "# TYPE mot_reid_jobs_total gauge",
    ]
    for status_name in ("queued", "running", "cancel_requested", "canceled", "failed", "completed"):
        lines.append(f'mot_reid_jobs_total{{status="{status_name}"}} {int(job_summary.get(status_name, 0))}')
    lines.extend(
        [
            "# HELP mot_reid_model_cache_entries Cached model objects.",
            "# TYPE mot_reid_model_cache_entries gauge",
            f'mot_reid_model_cache_entries{{type="detector"}} {int(cache_status.get("detectors", 0))}',
            f'mot_reid_model_cache_entries{{type="reid"}} {int(cache_status.get("reid_encoders", 0))}',
            f'mot_reid_model_cache_entries{{type="clip"}} {int(cache_status.get("clip_models", 0))}',
        ]
    )
    return "\n".join(lines) + "\n"


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            metrics = await asyncio.to_thread(_dashboard_metrics)
            await websocket.send_json(metrics)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return


def _dashboard_metrics():
    with runtime.pipeline_lock:
        return runtime.pipeline.dashboard_metrics()
