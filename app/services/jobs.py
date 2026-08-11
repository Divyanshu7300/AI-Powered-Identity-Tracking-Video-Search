from __future__ import annotations

import traceback
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Dict, List


JobRunner = Callable[[Callable[[Dict[str, Any]], None]], Dict[str, Any]]
logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProcessingJob:
    job_id: str
    source_name: str
    input_path: str
    output_path: str
    output_url: str | None = None
    uploaded_filename: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    progress: Dict[str, Any] = field(
        default_factory=lambda: {
            "stage": "queued",
            "percent": 0.0,
            "frames_processed": 0,
            "sampled_frames_processed": 0,
            "total_frames": None,
            "message": "Waiting for worker.",
        }
    )
    result: Dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None

    def public_dict(self, include_traceback: bool = False) -> Dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "source_name": self.source_name,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "output_url": self.output_url,
            "uploaded_filename": self.uploaded_filename,
            "metadata": self.metadata,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if include_traceback:
            payload["traceback"] = self.traceback
        return payload


class JobManager:
    def __init__(self, max_workers: int = 1, history_limit: int = 50) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tracking-job")
        self._history_limit = history_limit
        self._lock = RLock()
        self._jobs: Dict[str, ProcessingJob] = {}
        self._order: List[str] = []

    def submit(
        self,
        *,
        source_name: str,
        input_path: str,
        output_path: str,
        runner: JobRunner,
        output_url: str | None = None,
        uploaded_filename: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        job = ProcessingJob(
            job_id=uuid.uuid4().hex,
            source_name=source_name,
            input_path=input_path,
            output_path=output_path,
            output_url=output_url,
            uploaded_filename=uploaded_filename,
            metadata=metadata or {},
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._trim_locked()
        logger.info("tracking_job_queued", extra={"job_id": job.job_id, "source_name": source_name})
        self._executor.submit(self._run_job, job.job_id, runner)
        return job.public_dict()

    def get(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.public_dict() if job is not None else None

    def list(self, session_id: str | None = None) -> Dict[str, Any]:
        with self._lock:
            jobs = [
                self._jobs[job_id].public_dict()
                for job_id in reversed(self._order)
                if job_id in self._jobs and self._matches_session(self._jobs[job_id], session_id)
            ]
        return {"jobs": jobs}

    def summary(self, session_id: str | None = None) -> Dict[str, Any]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if self._matches_session(job, session_id)]
            queued = sum(1 for job in jobs if job.status == "queued")
            running = sum(1 for job in jobs if job.status == "running")
            cancel_requested = sum(1 for job in jobs if job.status == "cancel_requested")
            canceled = sum(1 for job in jobs if job.status == "canceled")
            failed = sum(1 for job in jobs if job.status == "failed")
            completed = sum(1 for job in jobs if job.status == "completed")
        return {
            "queued": queued,
            "running": running,
            "cancel_requested": cancel_requested,
            "canceled": canceled,
            "failed": failed,
            "completed": completed,
            "total": queued + running + cancel_requested + canceled + failed + completed,
        }

    def cancel(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == "queued":
                job.status = "canceled"
                job.finished_at = _now_iso()
                job.error = "Job canceled before processing started."
                job.progress = {
                    **job.progress,
                    "stage": "canceled",
                    "message": job.error,
                }
            elif job.status == "running":
                job.status = "cancel_requested"
                job.progress = {
                    **job.progress,
                    "stage": "cancel_requested",
                    "message": "Cancel requested. Waiting for worker checkpoint.",
                }
            return job.public_dict()

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {"queued", "running", "cancel_requested"}:
                return False
            self._jobs.pop(job_id, None)
            self._order = [current_id for current_id in self._order if current_id != job_id]
            return True

    def clear_session(self, session_id: str | None = None) -> int:
        with self._lock:
            matching = [
                job_id for job_id, job in self._jobs.items()
                if self._matches_session(job, session_id)
            ]
            for job_id in matching:
                job = self._jobs[job_id]
                if job.status in {"queued", "running"}:
                    job.status = "cancel_requested"
                self._jobs.pop(job_id, None)
                if job_id in self._order:
                    self._order.remove(job_id)
            return len(matching)

    def _run_job(self, job_id: str, runner: JobRunner) -> None:
        if not self._mark_started(job_id):
            return

        def progress_callback(update: Dict[str, Any]) -> None:
            self.update_progress(job_id, update)

        try:
            result = runner(progress_callback)
        except JobCancelled:
            logger.info("tracking_job_canceled", extra={"job_id": job_id})
            self._mark_canceled(job_id, "Job canceled.")
            return
        except Exception as exc:
            logger.exception("tracking_job_failed", extra={"job_id": job_id})
            self._mark_failed(job_id, str(exc), traceback.format_exc())
            return
        logger.info("tracking_job_completed", extra={"job_id": job_id})
        self._mark_completed(job_id, result)

    def update_progress(self, job_id: str, update: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == "cancel_requested":
                raise JobCancelled()
            job.progress = {**job.progress, **update}

    def _mark_started(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status == "canceled":
                return False
            job.status = "running"
            job.started_at = _now_iso()
            job.progress = {
                **job.progress,
                "stage": "starting",
                "message": "Starting video processing.",
            }
            return True

    def _mark_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.finished_at = _now_iso()
            job.result = result
            job.output_url = result.get("output_url") or job.output_url
            job.progress = {
                **job.progress,
                "stage": "completed",
                "percent": 100.0,
                "message": "Processing complete.",
            }

    def _mark_canceled(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "canceled"
            job.finished_at = _now_iso()
            job.error = message
            job.progress = {
                **job.progress,
                "stage": "canceled",
                "message": message,
            }

    def _mark_failed(self, job_id: str, error: str, error_traceback: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.finished_at = _now_iso()
            job.error = error
            job.traceback = error_traceback
            job.progress = {
                **job.progress,
                "stage": "failed",
                "message": error,
            }

    def _trim_locked(self) -> None:
        while len(self._order) > self._history_limit:
            old_id = self._order.pop(0)
            old = self._jobs.get(old_id)
            if old is not None and old.status in {"queued", "running", "cancel_requested"}:
                self._order.insert(0, old_id)
                return
            self._jobs.pop(old_id, None)

    @staticmethod
    def _matches_session(job: ProcessingJob, session_id: str | None) -> bool:
        if session_id is None:
            return True
        return str(job.metadata.get("session_id") or "default") == session_id
