from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np

from app.services.evidence_store import EvidenceStore
from app.services.memory_engine import VisionMemoryEngine
from app.services.clip_export import TrackClipExporter
from app.services import model_cache
from app.services.persistence import TrackPersistenceStore
from app.services.rag import VideoRAGAnswerer
from app.services.reid_index import ReIDEmbeddingIndex
from app.services.semantic_search import SemanticPersonSearchIndex
from tracking.tracker import MultiObjectTracker
from utils.image import crop_bbox
from utils.visualization import draw_tracked_objects


@dataclass
class PipelineConfig:
    detector_model: str = "yolov8n.pt"
    reid_weights: str | None = None
    reid_model_name: str = "osnet_x0_25"
    conf_threshold: float = 0.50
    match_threshold: float = 0.58
    appearance_weight: float = 0.35
    reid_match_threshold: float = 0.22
    reid_max_age: int = 900
    max_missed: int = 18
    min_hits: int = 2
    frame_stride: int = 1
    semantic_interval: int = 10
    memory_interval: int = 1
    tracker_timeline_limit: int = 180


class MOTReIDPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        source_name: str = "default",
    ) -> None:
        self.config = config or PipelineConfig()
        self.source_name = source_name
        self.detector = None
        self.detector_lock = None
        self.reid_encoder = None
        self.reid_lock = None
        self.tracker = self._new_tracker(source_name)
        self.evidence_store = EvidenceStore()
        self.memory_engine = VisionMemoryEngine(evidence_store=self.evidence_store)
        self.semantic_index = SemanticPersonSearchIndex(evidence_store=self.evidence_store)
        self.rag_answerer = VideoRAGAnswerer()
        self.persistence = TrackPersistenceStore()
        self.reid_index = ReIDEmbeddingIndex()
        self._reid_index_dirty = True
        self.clip_exporter = TrackClipExporter()

    def run_video(
        self,
        source: str,
        output_path: str = "data/output/tracked_video.mp4",
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
    ) -> Dict[str, object]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Video not found: {source}")

        source_name = source_path.stem
        if self.tracker.source_name != source_name:
            self.source_name = source_name
            self.tracker = self._new_tracker(source_name)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0 or fps > 120:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = _open_video_writer(
            output_path,
            fps / max(1, self.config.frame_stride),
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Unable to write output video: {output_path}")

        started_at = time.perf_counter()
        frames_processed = 0
        sampled_frames_processed = 0
        frames_with_tracks = 0
        max_track_id = 0
        last_progress_at = 0.0

        def report_progress(stage: str, message: str, force: bool = False) -> None:
            nonlocal last_progress_at
            if progress_callback is None:
                return
            now = time.perf_counter()
            if not force and now - last_progress_at < 0.75:
                return
            last_progress_at = now
            percent = 0.0
            if total_frames:
                percent = min(99.0, round((frames_processed / total_frames) * 100.0, 2))
            progress_callback(
                {
                    "stage": stage,
                    "message": message,
                    "percent": percent,
                    "frames_processed": frames_processed,
                    "sampled_frames_processed": sampled_frames_processed,
                    "total_frames": total_frames,
                    "frames_with_tracks": frames_with_tracks,
                }
            )

        report_progress("opening", "Video opened. Preparing output writer.", force=True)

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames_processed += 1
                if (frames_processed - 1) % max(1, self.config.frame_stride) != 0:
                    report_progress("processing", "Skipping frame according to stride.")
                    continue

                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))

                timestamp_seconds = frames_processed / max(fps, 1.0)
                result = self.process_frame(
                    frame,
                    tracker=self.tracker,
                    frame_index=frames_processed,
                    sampled_frame_index=sampled_frames_processed + 1,
                    timestamp_seconds=timestamp_seconds,
                    update_memory=True,
                )
                tracks = result["tracks"]
                sampled_frames_processed += 1
                if tracks:
                    frames_with_tracks += 1
                    max_track_id = max(max_track_id, max(int(track["track_id"]) for track in tracks))
                report_progress("processing", "Processing frames.")

                rendered = [
                    type(
                        "RenderedTrack",
                        (),
                        {
                            "bbox": track["bbox"],
                            "track_id": track["track_id"],
                            "confidence": track["confidence"],
                        },
                    )()
                    for track in tracks
                ]
                writer.write(draw_tracked_objects(frame, rendered, fps=result.get("fps")))
        finally:
            cap.release()
            writer.release()

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        report_progress("saving", "Saving memories and analytics.", force=True)
        current_memories = self.tracker.list_track_memories()
        for memory in current_memories:
            memory["episode"] = self.memory_engine.build_basic_summary(memory)
        self.persistence.save_run(
            source_name=source_name,
            source_path=str(source_path),
            output_path=output_path,
            memories=current_memories,
            track_registry=self.tracker.track_registry,
        )
        self._reid_index_dirty = True
        self._ensure_reid_index()
        memories = self.list_track_memories()
        report_progress("finalizing", "Finalizing processing result.", force=True)
        return {
            "source": str(source_path),
            "output_path": output_path,
            "frames_processed": frames_processed,
            "sampled_frames_processed": sampled_frames_processed,
            "frame_stride": max(1, self.config.frame_stride),
            "frames_with_tracks": frames_with_tracks,
            "max_track_id": max_track_id,
            "avg_fps": round(sampled_frames_processed / elapsed, 2),
            "source_fps": round(float(fps), 2),
            "track_memories": memories,
            "semantic_status": self.semantic_index.status(),
            "dashboard_metrics": self.dashboard_metrics(),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        tracker: MultiObjectTracker | None = None,
        frame_index: int | None = None,
        sampled_frame_index: int | None = None,
        timestamp_seconds: float | None = None,
        update_memory: bool = True,
    ) -> Dict[str, object]:
        started_at = time.perf_counter()
        tracker = tracker or self.tracker
        self._ensure_models()

        with self.detector_lock:
            self.detector.conf_threshold = self.config.conf_threshold
            detections = self.detector.detect_person(frame)
        detections = self._filter_detections(detections)
        crops = [crop_bbox(frame, detection["bbox"]) for detection in detections]
        valid_detections = []
        valid_crops = []
        for detection, crop in zip(detections, crops):
            if crop is None:
                continue
            valid_detections.append(detection)
            valid_crops.append(crop)

        if valid_crops:
            with self.reid_lock:
                batch = self.reid_encoder.encode_crops(valid_crops)
            embeddings = batch.embeddings
            valid_detections = [valid_detections[index] for index in batch.valid_indices]
        else:
            embeddings = np.empty((0, getattr(self.reid_encoder, "embedding_dim", 512)), dtype=np.float32)
            valid_detections = []

        visible_tracks = tracker.update(
            valid_detections,
            embeddings,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

        if update_memory and frame_index is not None and frame_index % max(1, self.config.memory_interval) == 0:
            self.memory_engine.update_tracks(frame, visible_tracks, frame_index, timestamp_seconds)

        track_dicts = [track.as_dict() for track in visible_tracks]
        semantic_index_frame = sampled_frame_index if sampled_frame_index is not None else frame_index
        if semantic_index_frame is not None and semantic_index_frame % max(1, self.config.semantic_interval) == 0:
            self.semantic_index.add_track_observations(
                frame,
                track_dicts,
                source_name=tracker.source_name,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )

        elapsed = max(time.perf_counter() - started_at, 1e-9)
        return {
            "frame_index": frame_index,
            "sampled_frame_index": sampled_frame_index,
            "timestamp_seconds": timestamp_seconds,
            "detections": valid_detections,
            "tracks": track_dicts,
            "fps": round(1.0 / elapsed, 2),
            "frames_processed": tracker.frame_index,
            "dashboard_metrics": tracker.dashboard_metrics(),
        }

    def search_person(self, image_bytes: bytes, top_k: int = 5) -> Dict[str, object]:
        self._ensure_reid()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"matches": [], "indexed_track_memories": len(self.tracker.track_registry), "message": "Invalid image."}

        query_image, query_crop_status = self._query_person_crop(image)
        with self.reid_lock:
            query_batch = self.reid_encoder.encode_crops([query_image])
        if query_batch.embeddings.size == 0:
            return {"matches": [], "indexed_track_memories": len(self.tracker.track_registry), "message": "Could not encode query image."}

        query = query_batch.embeddings[0]
        self._ensure_reid_index()
        indexed_result = self.reid_index.search(query, top_k=top_k)
        index_status = indexed_result["status"]
        message = indexed_result["message"]
        if index_status.get("skipped_dimension_mismatch"):
            message = (
                f"{message} Skipped {index_status['skipped_dimension_mismatch']} memories "
                "with incompatible Re-ID embedding dimensions."
            )
        return {
            "matches": indexed_result["matches"],
            "indexed_track_memories": len(self.list_track_memories()),
            "query_crop_status": query_crop_status,
            "reid_status": self.reid_status(),
            "image_index_status": index_status,
            "skipped_dimension_mismatch": index_status.get("skipped_dimension_mismatch", 0),
            "message": message,
        }

    def search_person_by_text(
        self,
        query: str,
        top_k: int = 5,
        start_time_seconds: float | None = None,
        end_time_seconds: float | None = None,
    ) -> Dict[str, object]:
        result = self.semantic_index.search(
            query,
            top_k=top_k,
            start_time_seconds=start_time_seconds,
            end_time_seconds=end_time_seconds,
        )
        memories_by_id = {memory["memory_id"]: memory for memory in self.list_track_memories()}
        enriched = []
        for match in result.get("matches", []):
            memory = memories_by_id.get(match["memory_id"], {})
            enriched.append({**memory, **match})
        result["matches"] = enriched
        result["rag"] = self.rag_answerer.answer(query, enriched)
        return result

    def list_track_memories(self) -> List[Dict[str, object]]:
        runtime_memories = self.tracker.list_track_memories()
        memories_by_id = {
            str(memory["memory_id"]): memory
            for memory in self.persistence.list_memories()
        }
        for memory in runtime_memories:
            memories_by_id[str(memory["memory_id"])] = memory
        memories = sorted(
            memories_by_id.values(),
            key=lambda item: (str(item.get("source_name", "")), int(item.get("track_id", 0))),
        )
        for memory in memories:
            memory["episode"] = self.memory_engine.build_basic_summary(memory)
        return memories

    def get_track_memory(self, memory_id: str) -> Dict[str, object]:
        decoded = memory_id.replace("%3A", ":")
        for memory in self.list_track_memories():
            if memory["memory_id"] == decoded:
                return memory
        persisted = self.persistence.get_memory(decoded)
        if persisted is not None:
            persisted["episode"] = self.memory_engine.build_basic_summary(persisted)
            return persisted
        raise KeyError(decoded)

    def export_track_clip(
        self,
        memory_id: str,
        padding_frames: int = 0,
    ) -> Dict[str, object]:
        memory = self.get_track_memory(memory_id)
        result = self.clip_exporter.export(
            memory,
            padding_frames=padding_frames,
        )
        memory["clip_url"] = result["clip_url"]
        self.persistence.update_memory_fields(
            str(memory["memory_id"]),
            {
                "clip_url": result["clip_url"],
                "clip_path": result["clip_path"],
            },
        )
        return result

    def dashboard_metrics(self) -> Dict[str, object]:
        memories = self.list_track_memories()
        sources = {memory["source_name"] for memory in memories}
        evidence_gallery = [
            {
                "memory_id": memory.get("memory_id"),
                "track_id": memory.get("track_id"),
                "evidence_url": memory.get("evidence_url"),
                "best_crop_url": memory.get("best_crop_url"),
                "crop_url": memory.get("crop_url"),
            }
            for memory in memories
            if memory.get("evidence_url") or memory.get("best_crop_url") or memory.get("crop_url")
        ]

        return {
            "overview": {
                "indexed_track_memories": len(memories),
                "sources_processed": len(sources),
                "persisted_track_memories": self.persistence.memory_count(),
                "persisted_sources": self.persistence.source_count(),
                "semantic_observations": self.semantic_index.status()["semantic_observations"],
                "active_tracks": len(self.tracker.tracks),
                "frames_processed": self.tracker.frame_index,
            },
            "tracker": self.tracker.dashboard_metrics(),
            "semantic_status": self.semantic_index.status(),
            "reid_status": self.reid_status(),
            "image_index_status": self.reid_index.status(),
            "model_cache": model_cache.status(),
            "evidence_gallery": evidence_gallery[-24:],
        }

    def reid_status(self) -> Dict[str, object]:
        if self.reid_encoder is None:
            return {
                "ready": False,
                "backend": None,
                "embedding_dim": None,
                "fallback_error": None,
            }
        return {
            "ready": True,
            "backend": getattr(self.reid_encoder, "backend", "unknown"),
            "embedding_dim": getattr(self.reid_encoder, "embedding_dim", None),
            "fallback_error": getattr(self.reid_encoder, "fallback_error", None),
        }

    def _query_person_crop(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, object]]:
        try:
            self._ensure_detector()
            with self.detector_lock:
                self.detector.conf_threshold = self.config.conf_threshold
                detections = self.detector.detect_person(image)
        except Exception as exc:
            return image, {
                "used_detector_crop": False,
                "message": f"Detector crop unavailable: {exc}",
            }

        if not detections:
            return image, {
                "used_detector_crop": False,
                "message": "No person detected in query image; encoded full image.",
            }

        best = max(
            detections,
            key=lambda detection: float(detection.get("confidence", 0.0)) * self._bbox_area(detection.get("bbox")),
        )
        crop = crop_bbox(image, best["bbox"])
        if crop is None:
            return image, {
                "used_detector_crop": False,
                "message": "Detected person crop was invalid; encoded full image.",
            }
        return crop, {
            "used_detector_crop": True,
            "bbox": list(map(int, best["bbox"])),
            "confidence": round(float(best.get("confidence", 0.0)), 4),
            "message": "Detected person crop encoded for Re-ID search.",
        }

    @staticmethod
    def _bbox_area(bbox) -> float:
        if not bbox:
            return 0.0
        x1, y1, x2, y2 = map(float, bbox)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _new_tracker(self, source_name: str) -> MultiObjectTracker:
        return MultiObjectTracker(
            max_missed=self.config.max_missed,
            min_hits=self.config.min_hits,
            match_threshold=self.config.match_threshold,
            appearance_weight=self.config.appearance_weight,
            reid_match_threshold=self.config.reid_match_threshold,
            reid_max_age=self.config.reid_max_age,
            timeline_limit=self.config.tracker_timeline_limit,
            source_name=source_name,
        )

    def _runtime_track_for_memory(self, memory: Dict[str, object]):
        memory_source = str(memory.get("source_name", ""))
        memory_track_id = int(memory.get("track_id", 0))
        track = self.tracker.track_registry.get(memory_track_id)
        if track is None:
            return None
        if str(getattr(track, "source_name", "")) != memory_source:
            return None
        return track

    def _ensure_reid_index(self) -> None:
        memories = self.list_track_memories()
        if not self._reid_index_dirty and self.reid_index.status()["memories_indexed"] == len(memories):
            return
        self.reid_index.rebuild(memories, self._embedding_candidates_for_memory)
        self._reid_index_dirty = False

    def _embedding_candidates_for_memory(self, memory: Dict[str, object]) -> np.ndarray | None:
        track = self._runtime_track_for_memory(memory)
        if track is not None and hasattr(track, "embedding_candidates"):
            return track.embedding_candidates()
        return self.persistence.load_embedding(str(memory["memory_id"]))

    def _ensure_models(self) -> None:
        self._ensure_detector()
        self._ensure_reid()

    def _ensure_detector(self) -> None:
        if self.detector is not None:
            return
        self.detector, self.detector_lock = model_cache.get_detector(
            self.config.detector_model,
            self.config.conf_threshold,
        )

    def _ensure_reid(self) -> None:
        if self.reid_encoder is not None:
            return
        self.reid_encoder, self.reid_lock = model_cache.get_reid_encoder(
            self.config.reid_weights,
            self.config.reid_model_name,
        )

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        if left.shape != right.shape:
            return 0.0
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-9:
            return 0.0
        return float(np.dot(left, right) / denom)

    @classmethod
    def _best_cosine(cls, query: np.ndarray, candidates: np.ndarray) -> float:
        candidate_array = np.asarray(candidates, dtype=np.float32)
        if candidate_array.ndim == 1:
            return cls._cosine(query, candidate_array)
        if candidate_array.size == 0:
            return 0.0
        compatible = [
            candidate
            for candidate in candidate_array
            if np.asarray(candidate).reshape(-1).shape == np.asarray(query).reshape(-1).shape
        ]
        if not compatible:
            return 0.0
        return max(cls._cosine(query, candidate) for candidate in compatible)

    @staticmethod
    def _embedding_dimensions_match(query: np.ndarray, candidates: np.ndarray) -> bool:
        query_size = int(np.asarray(query).reshape(-1).size)
        candidate_array = np.asarray(candidates)
        if candidate_array.ndim == 1:
            return int(candidate_array.reshape(-1).size) == query_size
        if candidate_array.ndim == 2:
            return int(candidate_array.shape[1]) == query_size
        return False


def _open_video_writer(output_path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
