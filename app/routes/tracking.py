from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from threading import Lock

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.services import model_cache, runtime
from app.services.pipeline import MOTReIDPipeline, PipelineConfig
from app.services.auth import require_auth


router = APIRouter(prefix="/tracking", tags=["tracking"], dependencies=[Depends(require_auth)])
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = int(os.getenv("MOT_REID_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024))) # 512 MB
LOCAL_VIDEO_ROOTS = (Path("data/users").resolve(), Path("test").resolve())
ALLOWED_DETECTOR_MODELS = {"yolov8n.pt", "yolov8s.pt"}
MAX_IMAGE_BYTES = int(os.getenv("MOT_REID_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
UPLOADS_PER_HOUR = int(os.getenv("MOT_REID_UPLOADS_PER_HOUR", "10"))
SESSION_QUOTA_BYTES = int(os.getenv("MOT_REID_SESSION_QUOTA_BYTES", str(2 * 1024 * 1024 * 1024)))
_upload_events: dict[str, list[tuple[float, int]]] = {}
_upload_lock = Lock()


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
    use_llm: bool = True



class ClipExportRequest(BaseModel):
    padding_frames: int = Field(default=0, ge=0, le=300)


class ModelWarmupRequest(BaseModel):
    detector: bool = True
    reid: bool = True
    clip: bool = False


class SemanticSettingsRequest(BaseModel):
    clip_enabled: bool


@router.post("/run")
def run_tracking(payload: TrackRequest, request: Request):
    session_id = _session_id(request)
    _check_upload_quota(session_id)
    source_path = Path(payload.source_path).resolve()
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {payload.source_path}")
    if source_path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES or not _is_allowed_local_video(source_path):
        raise HTTPException(status_code=400, detail="Source video must be inside data/input or test.")
    output_path = Path(payload.output_path)
    if not output_path.is_absolute():
        output_path = (_session_root(session_id) / "output" / output_path.name).resolve()
    else:
        output_path = output_path.resolve()
    if not _is_within(output_path, _session_root(session_id) / "output"):
        raise HTTPException(status_code=400, detail="Output path must be inside the session output directory.")

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
    namespace_root = _session_root(session_id)
    uploads_dir = namespace_root / "input/uploads"
    outputs_dir = namespace_root / "output"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    upload_id = uuid.uuid4().hex
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a supported video file (mp4, avi, mov, mkv, or webm).")
    if not file.content_type or not file.content_type.startswith("video/"):
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
    if not _is_valid_video_content(input_path):
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded content is not a readable video.")
    if _directory_size(namespace_root) > SESSION_QUOTA_BYTES:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Session storage quota exceeded.")
    _record_upload(session_id, bytes_written)

    config = runtime.snapshot_config(session_id)
    if detector_model:
        if detector_model not in ALLOWED_DETECTOR_MODELS:
            raise HTTPException(status_code=400, detail="Unsupported detector model.")
        config.detector_model = detector_model
    if frame_stride:
        config.frame_stride = max(1, int(frame_stride))
    return _submit_video_job(
        config,
        input_path,
        output_path,
        session_id=session_id,
        uploaded_filename=file.filename,
        output_url=f"/media/outputs/{output_path.name}",
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
            clip_model_name=config.semantic_model_name if payload.clip else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model warm-up failed: {exc}")


@router.get("/search/settings")
def get_search_settings(request: Request):
    return runtime.get_pipeline(_session_id(request)).semantic_index.status()


@router.post("/search/settings")
def update_search_settings(payload: SemanticSettingsRequest, request: Request):
    pipeline = runtime.get_pipeline(_session_id(request))
    pipeline.config.semantic_enable_clip = payload.clip_enabled
    pipeline.semantic_index.set_clip_enabled(payload.clip_enabled)
    return pipeline.semantic_index.status()


@router.post("/search/reindex")
def reindex_search(request: Request):
    """Refresh captions and vectors for the session's currently processed video."""
    pipeline = runtime.get_pipeline(_session_id(request))
    result = pipeline.reindex_current_video()
    if result.get("message"):
        raise HTTPException(status_code=409, detail=result["message"])
    return result


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
    if not _is_within(output_path, _session_root(session_id) / "output"):
        raise HTTPException(status_code=400, detail="Output path must be inside the session output directory.")

    return _submit_video_job(
        config,
        source_path,
        output_path,
        session_id=session_id,
        uploaded_filename=job.get("uploaded_filename"),
        output_url=job.get("output_url"),
    )


@router.delete("/jobs/{job_id:path}")
def delete_job(job_id: str, request: Request):
    existing = runtime.job_manager.get(job_id)
    if existing is not None:
        try:
            _ensure_job_session(existing, _session_id(request))
        except HTTPException:
            pass
        runtime.job_manager.delete(job_id)
    return {"deleted": True, "job_id": job_id}


@router.post("/search")
def search_person(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 5,
    mode: str = "hybrid",
):
    image_bytes = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Query image exceeds the configured size limit.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Query file must be an image.")
    return runtime.get_pipeline(_session_id(request)).search_person(image_bytes=image_bytes, top_k=top_k, mode=mode)


@router.post("/search/text")
def search_person_by_text(payload: TextSearchRequest, request: Request):
    return runtime.get_pipeline(_session_id(request)).search_person_by_text(
        query=payload.query,
        top_k=payload.top_k,
        start_time_seconds=payload.start_time_seconds,
        end_time_seconds=payload.end_time_seconds,
        use_llm=payload.use_llm,
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


@router.delete("/analytics/tracks/{memory_id:path}")
def delete_track_memory_endpoint(memory_id: str, request: Request):
    try:
        runtime.get_pipeline(_session_id(request)).delete_track_memory(memory_id)
        return {"deleted": True, "memory_id": memory_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/clips/{memory_id:path}")
def export_track_clip(
    memory_id: str,
    payload: Optional[ClipExportRequest] = Body(default_factory=ClipExportRequest),
    request: Request = None,
):
    try:
        padding_frames = payload.padding_frames if payload else 15
        return runtime.get_pipeline(_session_id(request)).export_track_clip(
            memory_id=memory_id,
            padding_frames=padding_frames,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Track memory not found: {memory_id}")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/reset")
def reset_session_data(request: Request):
    session_id = _session_id(request)
    runtime.reset_session(session_id)
    namespace_root = _session_root(session_id)
    for folder in namespace_root.iterdir() if namespace_root.exists() else []:
        if folder.is_dir():
            for item in folder.rglob("*"):
                if item.is_file() and not item.name.startswith("."):
                    item.unlink(missing_ok=True)
    return {
        "status": "reset",
        "session_id": session_id,
        "message": "Session data, video archives, and track memories reset successfully.",
    }


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
    source_display_name = uploaded_filename or source_path.stem
    def runner(progress_callback):
        source_label = Path(uploaded_filename or source_path.name).stem
        next_pipeline = MOTReIDPipeline(
            config=config,
            source_name=source_display_name,
            source_label=source_label,
            data_root=_session_root(session_id),
        )
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
        source_name=source_display_name,
        input_path=str(source_path),
        output_path=str(output_path),
        output_url=output_url,
        uploaded_filename=uploaded_filename,
        metadata={"config": asdict(config), "session_id": session_id},
        runner=runner,
    )


def _session_id(request: Request) -> str:
    return runtime.normalize_session_id(request.state.identity["sid"])


def _session_root(session_id: str) -> Path:
    return Path("data/users") / runtime.normalize_session_id(session_id)


def _check_upload_quota(session_id: str) -> None:
    now = time.time()
    with _upload_lock:
        events = [(stamp, size) for stamp, size in _upload_events.get(session_id, []) if now - stamp < 3600]
        if len(events) >= UPLOADS_PER_HOUR or sum(size for _, size in events) >= SESSION_QUOTA_BYTES:
            raise HTTPException(status_code=429, detail="Upload rate or session storage quota exceeded.")
        _upload_events[session_id] = events


def _record_upload(session_id: str, size: int) -> None:
    with _upload_lock:
        _upload_events.setdefault(session_id, []).append((time.time(), size))


def _is_valid_video_content(path: Path) -> bool:
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        return bool(capture.isOpened())
    finally:
        capture.release()


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


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
