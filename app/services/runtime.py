from __future__ import annotations

import re
from dataclasses import replace
from threading import RLock

from app.services.jobs import JobManager
from app.services.pipeline import MOTReIDPipeline, PipelineConfig


config = PipelineConfig(
    detector_model="yolov8n.pt",
    reid_model_name="osnet_ain_x1_0",
)

pipeline = MOTReIDPipeline(
    config,
    source_name="global",
    data_root="data/users/default",
)

pipeline_lock = RLock()
job_manager = JobManager(max_workers=1, history_limit=50)
_pipelines = {"default": pipeline}


def normalize_session_id(session_id: str | None) -> str:
    value = (session_id or "default").strip()
    value = re.sub(r"[^a-zA-Z0-9_.:-]", "_", value)
    return value[:80] or "default"


def get_pipeline(session_id: str | None = None) -> MOTReIDPipeline:
    normalized = normalize_session_id(session_id)
    with pipeline_lock:
        existing = _pipelines.get(normalized)
        if existing is not None:
            return existing
        next_pipeline = MOTReIDPipeline(
            config=replace(config),
            source_name=normalized,
            data_root=f"data/users/{normalized}",
        )
        _pipelines[normalized] = next_pipeline
        return next_pipeline


def session_count() -> int:
    with pipeline_lock:
        return len(_pipelines)


def snapshot_config(session_id: str | None = None) -> PipelineConfig:
    with pipeline_lock:
        return replace(get_pipeline(session_id).config)


def replace_pipeline(next_pipeline: MOTReIDPipeline, session_id: str | None = None) -> None:
    global pipeline
    normalized = normalize_session_id(session_id)
    with pipeline_lock:
        _pipelines[normalized] = next_pipeline
        if normalized == "default":
            pipeline = next_pipeline


def reset_session(session_id: str | None = None) -> MOTReIDPipeline:
    normalized = normalize_session_id(session_id)
    with pipeline_lock:
        existing = _pipelines.get(normalized)
        if existing is not None:
            existing.reset_all()
        else:
            existing = MOTReIDPipeline(
                config=replace(config),
                source_name=normalized,
                data_root=f"data/users/{normalized}",
            )
        job_manager.clear_session(normalized)
        _pipelines[normalized] = existing
        if normalized == "default":
            global pipeline
            pipeline = existing
        return existing
