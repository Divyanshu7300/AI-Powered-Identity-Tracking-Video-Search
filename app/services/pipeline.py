from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np

from app.services.evidence_store import EvidenceStore
from app.services.face_search import FaceSearchEngine
from app.services.memory_engine import VisionMemoryEngine
from app.services.clip_export import TrackClipExporter
from app.services import model_cache
from app.services.persistence import TrackPersistenceStore
from app.services.rag import VideoRAGAnswerer
from app.services.query_planner import PersonSearchPlanner, describe_relaxations
from app.services.reid_index import ReIDEmbeddingIndex
from app.services.semantic_search import SemanticPersonSearchIndex
from tracking.tracker import MultiObjectTracker
from utils.image import crop_bbox
from utils.visualization import draw_tracked_objects


@dataclass
class PipelineConfig:
    detector_model: str = "yolov8n.pt"
    reid_weights: str | None = None
    reid_model_name: str = field(default_factory=lambda: os.getenv("REID_MODEL_NAME", "osnet_ain_x1_0"))
    conf_threshold: float = 0.50
    match_threshold: float = 0.58
    # Identity appearance must outweigh position when people cross paths.
    appearance_weight: float = 0.65
    reid_match_threshold: float = 0.32
    reid_max_age: int = 900
    # Keep a confirmed track alive through a short occlusion instead of
    # assigning a fresh ID when the person becomes visible again.
    max_missed: int = 60
    min_hits: int = 2
    frame_stride: int = 1
    semantic_interval: int = 10
    semantic_enable_clip: bool = False
    semantic_representatives_per_track: int = 8
    semantic_minimum_frame_gap: int = 30
    semantic_evidence_frames_per_track: int = 3
    semantic_model_name: str = field(
        default_factory=lambda: os.getenv("SEMANTIC_MODEL_NAME", "google/siglip-base-patch16-224")
    )
    memory_interval: int = 1
    tracker_timeline_limit: int = 180
    face_interval: int = 5


class MOTReIDPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        source_name: str = "default",
        source_label: str | None = None,
        data_root: str = "data",
    ) -> None:
        self.config = config or PipelineConfig()
        self.source_name = source_name
        self.source_label = source_label or source_name
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.current_video_source: str | None = None
        self.detector = None
        self.detector_lock = None
        self.reid_encoder = None
        self.reid_lock = None
        self.tracker = self._new_tracker(source_name)
        self.evidence_store = EvidenceStore(self.data_root / "crops", self.data_root / "evidence")
        self.memory_engine = VisionMemoryEngine(evidence_store=self.evidence_store)
        self.semantic_index = SemanticPersonSearchIndex(
            persist_dir=self.data_root / "semantic_chroma",
            crops_dir=self.data_root / "crops",
            evidence_dir=self.data_root / "evidence",
            evidence_store=self.evidence_store,
            representatives_per_track=self.config.semantic_representatives_per_track,
            minimum_frame_gap=self.config.semantic_minimum_frame_gap,
            evidence_frames_per_track=self.config.semantic_evidence_frames_per_track,
            semantic_model_name=self.config.semantic_model_name,
        )
        self.semantic_index.set_clip_enabled(self.config.semantic_enable_clip)
        self.rag_answerer = VideoRAGAnswerer()
        self.query_planner = PersonSearchPlanner()
        self.persistence = TrackPersistenceStore(self.data_root / "mot_reid.sqlite3", self.data_root / "embeddings")
        self.reid_index = ReIDEmbeddingIndex(persist_dir=self.data_root / "reid_index")
        self._reid_index_dirty = True
        self._reid_index_source: str | None = None
        self.clip_exporter = TrackClipExporter(self.data_root / "clips")
        self.face_search = FaceSearchEngine()

    def run_video(
        self,
        source: str,
        output_path: str = "data/output/tracked_video.mp4",
        progress_callback: Callable[[Dict[str, object]], None] | None = None,
    ) -> Dict[str, object]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Video not found: {source}")

        source_name = self._source_name_for_path(source_path)
        self.current_video_source = source_name
        if self.tracker.source_name != source_name:
            self.source_name = source_name
            self.tracker = self._new_tracker(source_name)
        # Reprocessing a source replaces its searchable observations instead of
        # mixing old detections with the current run.
        self.evidence_store.clear_source(source_name)
        self.semantic_index.clear_source(source_name)

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

        batch_size = 8
        try:
            while True:
                batch_data = []
                while len(batch_data) < batch_size:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    frames_processed += 1
                    if (frames_processed - 1) % max(1, self.config.frame_stride) != 0:
                        report_progress("processing", "Skipping frame according to stride.")
                        continue

                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))

                    sampled_frames_processed += 1
                    timestamp_seconds = frames_processed / max(fps, 1.0)
                    batch_data.append(
                        {
                            "frame": frame,
                            "frame_index": frames_processed,
                            "sampled_frame_index": sampled_frames_processed,
                            "timestamp_seconds": timestamp_seconds,
                        }
                    )

                if not batch_data:
                    break

                report_progress(
                    "processing",
                    f"Analyzing frames through {frames_processed}"
                    + (f"/{total_frames}" if total_frames else "")
                    + ".",
                    force=True,
                )
                batch_results = self.process_frames_batch(
                    batch_data,
                    tracker=self.tracker,
                    update_memory=True,
                )

                for item, res in zip(batch_data, batch_results):
                    frame = item["frame"]
                    tracks = res["tracks"]
                    if tracks:
                        frames_with_tracks += 1
                        max_track_id = max(max_track_id, max(int(track["track_id"]) for track in tracks))

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
                    writer.write(draw_tracked_objects(frame, rendered, fps=res.get("fps")))

                progress_msg = f"Processed {frames_processed}" + (f"/{total_frames}" if total_frames else "") + f" frames ({max_track_id} tracks found)"
                report_progress("processing", progress_msg, force=True)
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
            "source_name": source_name,
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

    def process_frames_batch(
        self,
        frames_data: List[Dict[str, object]],
        tracker: MultiObjectTracker | None = None,
        update_memory: bool = True,
    ) -> List[Dict[str, object]]:
        if not frames_data:
            return []
        tracker = tracker or self.tracker
        self._ensure_models()

        frames = [item["frame"] for item in frames_data]
        with self.detector_lock:
            self.detector.conf_threshold = self.config.conf_threshold
            if hasattr(self.detector, "detect_objects_and_persons_batch"):
                raw_results = self.detector.detect_objects_and_persons_batch(frames)
                batch_detections = [res["persons"] for res in raw_results]
                batch_objects = [res["objects"] for res in raw_results]
            elif hasattr(self.detector, "detect_person_batch"):
                batch_detections = self.detector.detect_person_batch(frames)
                batch_objects = [[] for _ in frames]
            else:
                batch_detections = [self.detector.detect_person(frame) for frame in frames]
                batch_objects = [[] for _ in frames]

        all_crops = []
        crop_to_frame_map = []
        filtered_batch_detections = []

        for frame_idx, (item, detections) in enumerate(zip(frames_data, batch_detections)):
            frame = item["frame"]
            filtered_dets = self._filter_detections(detections)
            filtered_batch_detections.append(filtered_dets)
            for det_idx, det in enumerate(filtered_dets):
                crop = crop_bbox(frame, det["bbox"])
                if crop is not None:
                    all_crops.append(crop)
                    crop_to_frame_map.append((frame_idx, det_idx))

        if all_crops:
            with self.reid_lock:
                batch_embeddings = self.reid_encoder.encode_crops(all_crops)
            embeddings_by_frame_det = {}
            for crop_idx in range(len(batch_embeddings.valid_indices)):
                valid_orig_idx = batch_embeddings.valid_indices[crop_idx]
                f_idx, d_idx = crop_to_frame_map[valid_orig_idx]
                emb = batch_embeddings.embeddings[crop_idx]
                embeddings_by_frame_det[(f_idx, d_idx)] = emb
        else:
            embeddings_by_frame_det = {}

        results = []
        for frame_idx, item in enumerate(frames_data):
            frame = item["frame"]
            f_num = item["frame_index"]
            sf_num = item["sampled_frame_index"]
            ts = item["timestamp_seconds"]
            dets = filtered_batch_detections[frame_idx]

            valid_dets = []
            emb_list = []
            for d_idx, det in enumerate(dets):
                if (frame_idx, d_idx) in embeddings_by_frame_det:
                    valid_dets.append(det)
                    emb_list.append(embeddings_by_frame_det[(frame_idx, d_idx)])

            if emb_list:
                embeddings = np.vstack(emb_list)
            else:
                embeddings = np.empty((0, getattr(self.reid_encoder, "embedding_dim", 512)), dtype=np.float32)

            visible_tracks = tracker.update(
                valid_dets,
                embeddings,
                frame_index=f_num,
                timestamp_seconds=ts,
            )

            if f_num is not None and f_num % max(1, self.config.face_interval) == 0:
                self._attach_faces_to_tracks(frame, visible_tracks)

            if update_memory and f_num is not None and f_num % max(1, self.config.memory_interval) == 0:
                self.memory_engine.update_tracks(frame, visible_tracks, f_num, ts)

            track_dicts = [track.as_dict() for track in visible_tracks]
            semantic_index_frame = sf_num if sf_num is not None else f_num
            if semantic_index_frame is not None and semantic_index_frame % max(1, self.config.semantic_interval) == 0:
                frame_objs = batch_objects[frame_idx] if frame_idx < len(batch_objects) else None
                self.semantic_index.add_track_observations(
                    frame,
                    track_dicts,
                    source_name=tracker.source_name,
                    frame_index=f_num,
                    timestamp_seconds=ts,
                    frame_objects=frame_objs,
                )


            results.append(
                {
                    "frame_index": f_num,
                    "sampled_frame_index": sf_num,
                    "timestamp_seconds": ts,
                    "detections": valid_dets,
                    "tracks": track_dicts,
                    "frames_processed": tracker.frame_index,
                    "dashboard_metrics": tracker.dashboard_metrics(),
                }
            )

        return results

    def process_frame(
        self,
        frame: np.ndarray,
        tracker: MultiObjectTracker | None = None,
        frame_index: int | None = None,
        sampled_frame_index: int | None = None,
        timestamp_seconds: float | None = None,
        update_memory: bool = True,
    ) -> Dict[str, object]:
        res = self.process_frames_batch(
            [
                {
                    "frame": frame,
                    "frame_index": frame_index,
                    "sampled_frame_index": sampled_frame_index,
                    "timestamp_seconds": timestamp_seconds,
                }
            ],
            tracker=tracker,
            update_memory=update_memory,
        )
        return res[0] if res else {}


    def search_person(self, image_bytes: bytes, top_k: int = 5, mode: str = "hybrid") -> Dict[str, object]:
        if not self.current_video_source:
            return self._no_current_video_search_result("image")
        mode = mode if mode in {"face", "appearance", "hybrid"} else "hybrid"
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
        self._ensure_reid_index(source_name=self.current_video_source)
        indexed_result = self.reid_index.search(query, top_k=max(top_k * 3, 20))
        face_matches, face_message = self._search_faces(image, top_k=max(top_k * 3, 20))
        if mode == "face":
            matches = face_matches[:top_k]
            message = face_message
        elif mode == "appearance":
            matches = indexed_result["matches"][:top_k]
            message = indexed_result["message"]
        else:
            matches = self._combine_search_matches(indexed_result["matches"], face_matches, top_k)
            message = "Hybrid face + appearance search." if face_matches else f"Appearance fallback: {face_message}"
        index_status = indexed_result["status"]
        if index_status.get("skipped_dimension_mismatch"):
            message = (
                f"{message} Skipped {index_status['skipped_dimension_mismatch']} memories "
                "with incompatible Re-ID embedding dimensions."
            )
        return {
            "matches": matches,
            "indexed_track_memories": len(
                [memory for memory in self.list_track_memories() if memory.get("source_name") == self.current_video_source]
            ),
            "query_crop_status": query_crop_status,
            "reid_status": self.reid_status(),
            "image_index_status": index_status,
            "skipped_dimension_mismatch": index_status.get("skipped_dimension_mismatch", 0),
            "message": message,
            "search_mode": mode,
            "face_status": self.face_search.status(),
        }

    def search_person_by_text(
        self,
        query: str,
        top_k: int = 5,
        start_time_seconds: float | None = None,
        end_time_seconds: float | None = None,
        use_llm: bool = True,
    ) -> Dict[str, object]:
        if not self.current_video_source:
            return self._no_current_video_search_result("text")
        query_context = self.rag_answerer.rewrite_search_query(query) if use_llm else {
            "original_query": query, "search_query": query, "changed": False, "provider": "disabled"
        }
        plan = self.query_planner.build(query, query_context)
        result, strategy = self._run_adaptive_text_search(
            plan, top_k, start_time_seconds, end_time_seconds
        )
        memories_by_id = {
            memory["memory_id"]: memory
            for memory in self.list_track_memories()
            if memory.get("source_name") == self.current_video_source
        }
        enriched = []
        for match in result.get("matches", []):
            memory = memories_by_id.get(match["memory_id"], {})
            enriched_match = {**memory, **match}
            caption = str(enriched_match.get("caption") or "person match")
            enriched_match["match_reason"] = f"Matched: {caption}"
            enriched.append(enriched_match)
        result["matches"] = enriched
        result["query_context"] = query_context
        result["search_strategy"] = strategy
        if not enriched:
            result["no_result_guidance"] = self._no_result_guidance(plan)
        result["rag"] = self.rag_answerer.answer(query, enriched)
        return result

    def _run_adaptive_text_search(self, plan, top_k: int, start_time_seconds, end_time_seconds):
        """Run exact retrieval first, then relax only the weakest constraints."""
        for retrieval_pass in plan.passes:
            pass_matches = []
            for query_variant in retrieval_pass.queries:
                result = self.semantic_index.search(
                    query_variant,
                    top_k=max(10, top_k * 2),
                    start_time_seconds=start_time_seconds,
                    end_time_seconds=end_time_seconds,
                    source_name=self.current_video_source,
                    constraints_query=plan.original_query,
                    relaxed_fields=retrieval_pass.relaxed_fields,
                    allow_keyword_fallback=False,
                )
                for rank, match in enumerate(result.get("matches", []), start=1):
                    pass_matches.append({**match, "_rank": rank, "_query_variant": query_variant})
            if pass_matches:
                fused = self._fuse_text_matches(pass_matches, retrieval_pass, top_k)
                return {
                    "query": plan.original_query,
                    "matches": fused,
                    "indexed_semantic_observations": result.get("indexed_semantic_observations", 0),
                    "status": result.get("status", self.semantic_index.status()),
                    "retrieval_backend": result.get("retrieval_backend", "memory"),
                    "message": f"{retrieval_pass.label}: {len(fused)} ranked track matches.",
                }, {
                    "mode": "exact" if retrieval_pass.name == "exact" else "possible",
                    "label": retrieval_pass.label,
                    "relaxed_constraints": describe_relaxations(retrieval_pass.relaxed_fields),
                    "ambiguities": list(plan.ambiguities),
                }
        return {
            "query": plan.original_query, "matches": [], "indexed_semantic_observations": len(self.semantic_index.observations),
            "status": self.semantic_index.status(), "retrieval_backend": "adaptive", "message": "No exact or relaxed video evidence matched.",
        }, {"mode": "none", "label": "No match", "relaxed_constraints": [], "ambiguities": list(plan.ambiguities)}

    @staticmethod
    def _fuse_text_matches(matches: List[Dict[str, object]], retrieval_pass, top_k: int) -> List[Dict[str, object]]:
        grouped: Dict[str, Dict[str, object]] = {}
        for item in matches:
            memory_id = str(item.get("memory_id") or "")
            if not memory_id:
                continue
            contribution = 1.0 / (60.0 + int(item.get("_rank", 1)))
            current = grouped.get(memory_id)
            if current is None:
                current = {**item, "_fusion_score": contribution, "_variants": [item.get("_query_variant")]}
                grouped[memory_id] = current
            else:
                current["_fusion_score"] += contribution
                current["_variants"].append(item.get("_query_variant"))
                if float(item.get("score", 0.0)) > float(current.get("score", 0.0)):
                    preserved_score = current["_fusion_score"]
                    variants = current["_variants"]
                    current.update(item)
                    current["_fusion_score"] = preserved_score
                    current["_variants"] = variants
        fused = []
        relaxed_constraints = describe_relaxations(retrieval_pass.relaxed_fields)
        for item in grouped.values():
            item.pop("_rank", None)
            item["retrieval_strategy"] = retrieval_pass.label
            item["relaxed_constraints"] = relaxed_constraints
            item["query_variants"] = list(dict.fromkeys(str(value) for value in item.pop("_variants", []) if value))
            item["fusion_score"] = round(float(item.pop("_fusion_score", 0.0)), 5)
            fused.append(item)
        return sorted(fused, key=lambda item: (float(item["fusion_score"]), float(item.get("score", 0.0))), reverse=True)[:top_k]

    @staticmethod
    def _no_result_guidance(plan) -> Dict[str, object]:
        suggestions = ["Try a shorter visual description; the system will preserve only attributes visible in the footage."]
        if plan.intent.required_objects:
            suggestions.append("Try searching without the carried object; bags, phones, and umbrellas are often occluded.")
        if plan.intent.upper_color or plan.intent.lower_color:
            suggestions.append("Try broad colour wording such as dark, light, blue, or black; lighting can shift clothing labels.")
        if plan.intent.horizontal_zone or plan.intent.vertical_zone:
            suggestions.append("Remove the screen location or add an approximate time range; position changes throughout a video.")
        return {"reason": "No evidence passed the exact and controlled relaxed retrieval stages.", "suggestions": suggestions}

    def reindex_current_video(self) -> Dict[str, object]:
        if not self.current_video_source:
            return self._no_current_video_search_result("reindex")
        return self.semantic_index.reindex_source(self.current_video_source)

    def reset_all(self) -> None:
        """Purge all in-memory trackers, stored evidence, SQLite database, ChromaDB vectors, and video archives."""
        self.tracker = self._new_tracker(self.source_name)
        self.current_video_source = None
        self.evidence_store.clear_all()
        self.semantic_index.clear_all()
        self.persistence.clear_all()

        dirs_to_clean = [self.data_root / name for name in ("output", "input/uploads", "clips", "crops", "evidence", "embeddings")]
        for folder in dirs_to_clean:
            if folder.exists():
                for item in folder.iterdir():
                    if item.is_file() and not item.name.startswith("."):
                        try:
                            item.unlink(missing_ok=True)
                        except Exception:
                            pass

    def list_track_memories(self) -> List[Dict[str, object]]:
        runtime_memories = self.tracker.list_track_memories()
        memories_by_id = {
            str(memory["memory_id"]): memory
            for memory in self.persistence.list_memories()
        }
        for memory in runtime_memories:
            memory_id = str(memory["memory_id"])
            # Runtime data is fresher for tracking fields, while persisted data
            # contains source/output paths and previously exported clip details.
            memories_by_id[memory_id] = {**memories_by_id.get(memory_id, {}), **memory}
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

    def delete_track_memory(self, memory_id: str) -> bool:
        decoded = memory_id.replace("%3A", ":")
        try:
            track_id = int(decoded.split(":")[-1])
            self.tracker.track_registry.pop(track_id, None)
            self.tracker.tracks = [t for t in self.tracker.tracks if getattr(t, "track_id", None) != track_id]
        except Exception:
            pass
        self.evidence_store.clear_track(decoded)
        self.persistence.delete_memory(decoded)
        self._reid_index_dirty = True
        return True

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

    def _attach_faces_to_tracks(self, frame: np.ndarray, tracks: List[object]) -> None:
        faces = self.face_search.extract(frame)
        for face in faces:
            fx1, fy1, fx2, fy2 = face["bbox"]
            center_x, center_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            for track in tracks:
                x1, y1, x2, y2 = map(float, track.bbox)
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    track.add_face_sample(face["embedding"])
                    break

    def _search_faces(self, image: np.ndarray, top_k: int) -> tuple[List[Dict[str, object]], str]:
        faces = self.face_search.extract(image)
        if not faces:
            return [], "No high-quality face found in the target image."
        query = max(faces, key=lambda item: item["quality"])["embedding"]
        matches = []
        for track in self.tracker.track_registry.values():
            if track.source_name != self.current_video_source or not track.face_gallery:
                continue
            score = max(float(np.dot(query, sample)) for sample in track.face_gallery)
            memory = next((item for item in self.list_track_memories() if item["memory_id"] == f"{track.source_name}:{track.track_id}"), {})
            matches.append({**memory, "score": round(score, 4), "similarity": round(score, 4), "face_score": round(score, 4), "score_breakdown": {"face": round(score, 4)}})
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k], "Face template search."

    @staticmethod
    def _combine_search_matches(appearance_matches, face_matches, top_k: int) -> List[Dict[str, object]]:
        combined = {}
        for match in appearance_matches:
            appearance = max(0.0, float(match.get("score", 0.0)))
            combined[match["memory_id"]] = {**match, "appearance_score": round(appearance, 4)}
        for match in face_matches:
            current = combined.get(match["memory_id"], {})
            combined[match["memory_id"]] = {**current, **match, "face_score": float(match["score"])}
        ranked = []
        for match in combined.values():
            face = match.get("face_score")
            appearance = match.get("appearance_score")
            score = 0.70 * face + 0.30 * appearance if face is not None and appearance is not None else (face if face is not None else appearance)
            match["score"] = round(float(score or 0), 4)
            match["similarity"] = match["score"]
            match["score_breakdown"] = {"face": face, "appearance": appearance}
            ranked.append(match)
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]

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
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _filter_detections(self, detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Keep well-formed person detections before crops are generated.

        The detector is responsible for confidence and person-shape filtering.
        This second, lightweight check prevents malformed boxes from reaching the
        cropper and tracker when a detector backend returns unexpected data.
        """
        filtered: List[Dict[str, object]] = []
        for detection in detections or []:
            bbox = detection.get("bbox")
            if self._bbox_area(bbox) <= 0:
                continue
            try:
                confidence = float(detection.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(confidence) or confidence < self.config.conf_threshold:
                continue
            filtered.append(detection)
        return filtered

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
            source_label=self.source_label,
        )

    @staticmethod
    def _source_name_for_path(source_path: Path) -> str:
        """Create a stable, collision-resistant ID for a local video path."""
        resolved = str(source_path.resolve())
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
        return f"{source_path.stem}-{digest}"

    def _runtime_track_for_memory(self, memory: Dict[str, object]):
        memory_source = str(memory.get("source_name", ""))
        memory_track_id = int(memory.get("track_id", 0))
        track = self.tracker.track_registry.get(memory_track_id)
        if track is None:
            return None
        if str(getattr(track, "source_name", "")) != memory_source:
            return None
        return track

    def _no_current_video_search_result(self, search_type: str) -> Dict[str, object]:
        label = "Image Search" if search_type == "image" else "Text Search"
        return {
            "matches": [],
            "message": f"Run a video first. {label} only searches the current video.",
        }

    def _ensure_reid_index(self, source_name: str | None = None) -> None:
        memories = self.list_track_memories()
        if source_name:
            memories = [memory for memory in memories if memory.get("source_name") == source_name]
        if (
            not self._reid_index_dirty
            and self._reid_index_source == source_name
            and self.reid_index.status()["memories_indexed"] == len(memories)
        ):
            return
        self.reid_index.rebuild(memories, self._embedding_candidates_for_memory)
        self._reid_index_dirty = False
        self._reid_index_source = source_name

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
