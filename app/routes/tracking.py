from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.services import model_cache, runtime
from app.services.pipeline import MOTReIDPipeline, PipelineConfig


router = APIRouter(prefix="/tracking", tags=["tracking"])
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = int(os.getenv("MOT_REID_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
LOCAL_VIDEO_ROOTS = (Path("data/input").resolve(), Path("test").resolve())
LOCAL_OUTPUT_ROOT = Path("data/output").resolve()


class TrackRequest(BaseModel):
    source_path: str
    output_path: str = "data/output/tracked_video.mp4"
    conf_threshold: float = 0.50
    match_threshold: float = 0.58


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    start_time_seconds: Optional[float] = None
    end_time_seconds: Optional[float] = None



class ClipExportRequest(BaseModel):
    padding_frames: int = Field(default=0, ge=0, le=300)


class ModelWarmupRequest(BaseModel):
    detector: bool = True
    reid: bool = True
    clip: bool = False


@router.post("/run")
def run_tracking(payload: TrackRequest, request: Request):
    session_id = _session_id(request)
    source_path = Path(payload.source_path).resolve()
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {payload.source_path}")
    if source_path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES or not _is_allowed_local_video(source_path):
        raise HTTPException(status_code=400, detail="Source video must be inside data/input or test.")
    output_path = Path(payload.output_path).resolve()
    if not _is_within(output_path, LOCAL_OUTPUT_ROOT):
        raise HTTPException(status_code=400, detail="Output path must be inside data/output.")

    config = runtime.snapshot_config(session_id)
    config.conf_threshold = payload.conf_threshold
    config.match_threshold = payload.match_threshold
    return _submit_video_job(config, source_path, output_path, session_id=session_id)


@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    detector_model: Optional[str] = Form(None),
    frame_stride: Optional[int] = Form(None),
):
    session_id = _session_id(request)
    uploads_dir = Path("data/input/uploads")
    outputs_dir = Path("data/output")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    upload_id = uuid.uuid4().hex
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a supported video file (mp4, avi, mov, mkv, or webm).")
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must have a video content type.")
    input_path = uploads_dir / f"{upload_id}{suffix}"
    output_path = outputs_dir / f"{upload_id}_tracked.mp4"
    bytes_written = 0
    try:
        with input_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the configured upload size limit.")
                buffer.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    config = runtime.snapshot_config(session_id)
    if detector_model:
        config.detector_model = detector_model
    if frame_stride:
        config.frame_stride = max(1, int(frame_stride))
    return _submit_video_job(
        config,
        input_path,
        output_path,
        session_id=session_id,
        uploaded_filename=file.filename,
        output_url=f"/outputs/{output_path.name}",
    )


@router.post("/models/warmup")
def warmup_models(payload: ModelWarmupRequest, request: Request):
    config = runtime.snapshot_config(_session_id(request))
    try:
        return model_cache.warmup(
            detector_model=config.detector_model if payload.detector else None,
            detector_conf_threshold=config.conf_threshold,
            reid_weights=config.reid_weights if payload.reid else None,
            reid_model_name=config.reid_model_name if payload.reid else None,
            clip_model_name="openai/clip-vit-base-patch32" if payload.clip else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model warm-up failed: {exc}")


@router.get("/jobs")
def list_jobs(request: Request):
    return runtime.job_manager.list(_session_id(request))


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job = runtime.job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    _ensure_job_session(job, _session_id(request))
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    existing = runtime.job_manager.get(job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    _ensure_job_session(existing, _session_id(request))
    job = runtime.job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request):
    session_id = _session_id(request)
    job = runtime.job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    _ensure_job_session(job, session_id)
    if job["status"] in {"queued", "running", "cancel_requested"}:
        raise HTTPException(status_code=409, detail="Only finished jobs can be retried.")

    metadata = job.get("metadata") or {}
    config_data = metadata.get("config") or {}
    config = PipelineConfig(**config_data)
    source_path = Path(str(job["input_path"])).resolve()
    if not source_path.exists() or not (source_path.suffix.lower() in ALLOWED_VIDEO_SUFFIXES and _is_allowed_local_video(source_path)):
        raise HTTPException(status_code=400, detail=f"Invalid or unauthorized source video path: {source_path}")

    output_path = Path(str(job["output_path"])).resolve()
    if not _is_within(output_path, LOCAL_OUTPUT_ROOT):
        raise HTTPException(status_code=400, detail="Output path must be inside data/output.")

    return _submit_video_job(
        config,
        source_path,
        output_path,
        session_id=session_id,
        uploaded_filename=job.get("uploaded_filename"),
        output_url=job.get("output_url"),
    )


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str, request: Request):
    existing = runtime.job_manager.get(job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    _ensure_job_session(existing, _session_id(request))
    if not runtime.job_manager.delete(job_id):
        raise HTTPException(status_code=409, detail="Job not found or still active.")
    return {"deleted": True, "job_id": job_id}


@router.post("/search")
def search_person(request: Request, file: UploadFile = File(...), top_k: int = 5):
    return runtime.get_pipeline(_session_id(request)).search_person(image_bytes=file.file.read(), top_k=top_k)


@router.post("/search/text")
def search_person_by_text(payload: TextSearchRequest, request: Request):
    return runtime.get_pipeline(_session_id(request)).search_person_by_text(
        query=payload.query,
        top_k=payload.top_k,
        start_time_seconds=payload.start_time_seconds,
        end_time_seconds=payload.end_time_seconds,
    )


@router.get("/tracks")
def get_track_memories(request: Request):
    return {"track_memories": _list_track_memories(_session_id(request))}


@router.get("/analytics/dashboard")
def analytics_dashboard(request: Request):
    return _dashboard_metrics(_session_id(request))


@router.get("/analytics/tracks")
def analytics_tracks(request: Request):
    return {"track_memories": _list_track_memories(_session_id(request))}


@router.get("/analytics/tracks/{memory_id:path}")
def analytics_track(memory_id: str, request: Request):
    try:
        return runtime.get_pipeline(_session_id(request)).get_track_memory(memory_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Track memory not found: {memory_id}")


@router.post("/clips/{memory_id:path}")

def export_track_clip(memory_id: str, payload: ClipExportRequest, request: Request):
    try:
        return runtime.get_pipeline(_session_id(request)).export_track_clip(
            memory_id=memory_id,
            padding_frames=payload.padding_frames,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Track memory not found: {memory_id}")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/health")
def health(request: Request):
    session_id = _session_id(request)
    return {
        "status": "ok",
        "session_id": session_id,
        "sessions": runtime.session_count(),
        "jobs": runtime.job_manager.summary(session_id),
        "model_cache": model_cache.status(),
    }


def _list_track_memories(session_id: str):
    return runtime.get_pipeline(session_id).list_track_memories()


def _dashboard_metrics(session_id: str):
    metrics = runtime.get_pipeline(session_id).dashboard_metrics()
    metrics["session_id"] = session_id
    return metrics



def _submit_video_job(
    config,
    source_path: Path,
    output_path: Path,
    session_id: str,
    uploaded_filename: str | None = None,
    output_url: str | None = None,
):
    def runner(progress_callback):
        next_pipeline = MOTReIDPipeline(config=config, source_name=source_path.stem)
        result = next_pipeline.run_video(
            source=str(source_path),
            output_path=str(output_path),
            progress_callback=progress_callback,
        )
        if uploaded_filename:
            result["uploaded_filename"] = uploaded_filename
        if output_url:
            result["output_url"] = output_url
        runtime.replace_pipeline(next_pipeline, session_id)
        return result

    return runtime.job_manager.submit(
        source_name=source_path.stem,
        input_path=str(source_path),
        output_path=str(output_path),
        output_url=output_url,
        uploaded_filename=uploaded_filename,
        metadata={"config": asdict(config), "session_id": session_id},
        runner=runner,
    )


def _session_id(request: Request) -> str:
    return runtime.normalize_session_id(
        request.headers.get("x-session-id") or request.query_params.get("session_id")
    )


def _ensure_job_session(job: dict, session_id: str) -> None:
    job_session_id = str((job.get("metadata") or {}).get("session_id") or "default")
    if job_session_id != session_id:
        raise HTTPException(status_code=404, detail="Job not found in this session.")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_allowed_local_video(source_path: Path) -> bool:
    return any(_is_within(source_path, root) for root in LOCAL_VIDEO_ROOTS)
