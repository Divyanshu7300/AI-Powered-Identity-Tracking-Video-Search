from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.matcher import hungarian_match
from tracking.matcher import compute_iou
from tracking.matcher import cosine_distance
from tracking.matcher import normalized_center_distance
from tracking.matcher import shape_change_cost


@dataclass
class Track:
    track_id: int
    bbox: Sequence[float]
    confidence: float
    embedding: np.ndarray
    source_name: str = "default"
    hits: int = 1
    age: int = 0
    missed: int = 0
    first_frame: int = 0
    last_frame: int = 0
    best_confidence: float = 0.0
    velocity: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    metadata: Dict[str, object] = field(default_factory=dict)
    timeline: List[Dict[str, object]] = field(default_factory=list)
    embedding_gallery: List[np.ndarray] = field(default_factory=list)
    timeline_limit: int = 180

    def __post_init__(self) -> None:
        self.best_confidence = max(float(self.confidence), float(self.best_confidence))
        self.embedding = self._normalized_embedding(self.embedding)
        self._add_embedding_sample(self.embedding)

    def record_sighting(
        self,
        frame_index: int,
        bbox: Sequence[float],
        confidence: float,
        timestamp_seconds: float | None = None,
    ) -> None:
        self.last_frame = frame_index
        self.best_confidence = max(self.best_confidence, float(confidence))
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        self.timeline.append(
            {
                "frame_index": int(frame_index),
                "timestamp_seconds": round(float(timestamp_seconds), 2) if timestamp_seconds is not None else None,
                "bbox": list(map(int, bbox)),
                "center": [round(cx, 2), round(cy, 2)],
                "velocity": [round(float(value), 2) for value in self.velocity],
                "confidence": round(float(confidence), 4),
            }
        )
        self.timeline = self.timeline[-max(1, int(self.timeline_limit)):]

    def update(
        self,
        bbox: Sequence[float],
        confidence: float,
        embedding: np.ndarray,
        frame_index: int,
        timestamp_seconds: float | None = None,
    ) -> None:
        previous_bbox = np.asarray(self.bbox, dtype=np.float32)
        next_bbox = np.asarray(bbox, dtype=np.float32)
        displacement = next_bbox - previous_bbox
        velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.missed > 1:
            velocity *= 0.45
        self.velocity = tuple((0.75 * velocity + 0.25 * displacement).tolist())
        self.bbox = bbox
        self.confidence = confidence
        self.embedding = 0.72 * self.embedding + 0.28 * embedding
        self.embedding = self._normalized_embedding(self.embedding)
        self._add_embedding_sample(embedding)
        self.hits += 1
        self.age += 1
        self.missed = 0
        self.record_sighting(
            frame_index=frame_index,
            bbox=bbox,
            confidence=confidence,
            timestamp_seconds=timestamp_seconds,
        )

    def mark_missed(self) -> None:
        self.age += 1
        self.missed += 1

    def predicted_bbox(self) -> Sequence[float]:
        steps = min(max(self.missed, 0), 6)
        bbox = np.asarray(self.bbox, dtype=np.float32)
        velocity = np.asarray(self.velocity, dtype=np.float32)
        predicted = bbox + velocity * steps
        if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
            return self.bbox
        return tuple(predicted.tolist())

    def as_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "bbox": list(map(int, self.bbox)),
            "confidence": round(float(self.confidence), 4),
            "hits": self.hits,
            "age": self.age,
            "missed": self.missed,
            "source_name": self.source_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "best_confidence": round(float(self.best_confidence), 4),
            "timeline": self.timeline[-12:],
            "crop_url": self.metadata.get("crop_url"),
            "best_crop_url": self.metadata.get("best_crop_url"),
            "evidence_url": self.metadata.get("evidence_url"),
        }

    def embedding_candidates(self) -> np.ndarray:
        candidates = list(self.embedding_gallery)
        candidates.append(self.embedding)
        return np.vstack([self._normalized_embedding(candidate) for candidate in candidates])

    def _add_embedding_sample(self, embedding: np.ndarray, max_samples: int = 8) -> None:
        sample = self._normalized_embedding(embedding)
        if sample.size == 0:
            return
        if not self.embedding_gallery:
            self.embedding_gallery.append(sample)
            return

        similarities = [
            float(np.dot(sample, self._normalized_embedding(existing)))
            for existing in self.embedding_gallery
        ]
        if max(similarities) < 0.985 or len(self.embedding_gallery) < 3:
            self.embedding_gallery.append(sample)
        elif float(self.confidence) >= self.best_confidence:
            self.embedding_gallery[-1] = sample
        self.embedding_gallery = self.embedding_gallery[-max(1, int(max_samples)):]

    @staticmethod
    def _normalized_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


class MultiObjectTracker:
    def __init__(
        self,
        max_missed: int = 18,
        min_hits: int = 2,
        match_threshold: float = 0.58,
        appearance_weight: float = 0.35,
        reid_match_threshold: float = 0.22,
        reid_max_age: int = 900,
        timeline_limit: int = 180,
        source_name: str = "default",
    ) -> None:
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.match_threshold = match_threshold
        self.appearance_weight = appearance_weight
        self.reid_match_threshold = reid_match_threshold
        self.reid_max_age = reid_max_age
        self.timeline_limit = max(1, int(timeline_limit))
        self.source_name = source_name
        self.tracks: List[Track] = []
        self.track_registry: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_index = 0

    def _visible_tracks(self) -> List[Track]:
        return [
            track
            for track in self.tracks
            if track.missed == 0 and track.hits >= self.min_hits and track.track_id > 0
        ]

    def _promote_confirmed_tracks(self) -> None:
        for track in self.tracks:
            if track.track_id > 0 or track.hits < self.min_hits:
                continue
            track.track_id = self.next_track_id
            self.track_registry[track.track_id] = track
            self.next_track_id += 1

    def _deduplicate_live_tracks(self, iou_threshold: float = 0.82) -> None:
        if len(self.tracks) < 2:
            return

        keep = [True] * len(self.tracks)
        for left_idx, left in enumerate(self.tracks):
            if not keep[left_idx] or left.missed != 0:
                continue
            for right_idx in range(left_idx + 1, len(self.tracks)):
                right = self.tracks[right_idx]
                if not keep[right_idx] or right.missed != 0:
                    continue
                if compute_iou(left.bbox, right.bbox) < iou_threshold:
                    continue

                left_score = (left.hits, left.best_confidence, -left.track_id)
                right_score = (right.hits, right.best_confidence, -right.track_id)
                if right_score > left_score:
                    keep[left_idx] = False
                    break
                keep[right_idx] = False

        self.tracks = [track for track, should_keep in zip(self.tracks, keep) if should_keep]

    def _is_duplicate_detection(self, detection, iou_threshold: float = 0.72) -> bool:
        return any(
            track.missed == 0 and compute_iou(track.bbox, detection["bbox"]) >= iou_threshold
            for track in self.tracks
        )

    def _inactive_tracks(self) -> List[Track]:
        active_ids = {track.track_id for track in self.tracks}
        candidates = []
        for track in self.track_registry.values():
            if track.track_id in active_ids:
                continue
            if track.source_name != self.source_name:
                continue
            if track.hits < self.min_hits:
                continue
            if self.frame_index - track.last_frame > self.reid_max_age:
                continue
            candidates.append(track)
        return candidates

    def _mark_exited_tracks(self, tracks: List[Track]) -> None:
        for track in tracks:
            if track.track_id <= 0:
                continue
            events = track.metadata.setdefault("timeline_events", [])
            if events and events[-1].get("type") == "exited":
                continue
            timestamp = track.metadata.get("last_seen_timestamp_seconds")
            events.append(
                {
                    "type": "exited",
                    "label": "exited",
                    "frame_index": int(track.last_frame),
                    "timestamp_seconds": timestamp,
                }
            )
            track.metadata["timeline_events"] = events[-40:]

    def _reactivate_inactive_tracks(
        self,
        detections,
        embeddings: np.ndarray,
        det_indices: List[int],
        timestamp_seconds: float | None = None,
    ) -> set[int]:
        inactive_tracks = self._inactive_tracks()
        if not inactive_tracks or not det_indices:
            return set()

        costs = np.full((len(inactive_tracks), len(det_indices)), 1e6, dtype=np.float32)
        for track_idx, track in enumerate(inactive_tracks):
            for local_det_idx, det_idx in enumerate(det_indices):
                detection = detections[det_idx]
                appearance_cost = cosine_distance(track.embedding, embeddings[det_idx])
                if appearance_cost > self.reid_match_threshold:
                    continue

                center_cost = min(1.0, normalized_center_distance(track.bbox, detection["bbox"]))
                shape_cost = shape_change_cost(track.bbox, detection["bbox"])
                if shape_cost > 0.9 and appearance_cost > self.reid_match_threshold * 0.65:
                    continue

                # Old tracks may reappear far away, so appearance dominates here.
                costs[track_idx, local_det_idx] = (
                    0.82 * appearance_cost
                    + 0.10 * shape_cost
                    + 0.08 * center_cost
                )

        row_indices, col_indices = linear_sum_assignment(costs)
        reactivated_det_indices = set()
        for row_idx, col_idx in zip(row_indices.tolist(), col_indices.tolist()):
            if costs[row_idx, col_idx] > self.reid_match_threshold:
                continue

            track = inactive_tracks[row_idx]
            det_idx = det_indices[col_idx]
            detection = detections[det_idx]
            track.update(
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                embedding=embeddings[det_idx],
                frame_index=self.frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            self.tracks.append(track)
            reactivated_det_indices.add(det_idx)

        return reactivated_det_indices

    def update(
        self,
        detections,
        embeddings: np.ndarray,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> List[Track]:
        self.frame_index = self.frame_index + 1 if frame_index is None else frame_index

        for track in self.tracks:
            track.mark_missed()

        match_result = hungarian_match(
            self.tracks,
            detections,
            embeddings,
            max_cost=self.match_threshold,
            appearance_weight=self.appearance_weight,
        )

        for track_idx, det_idx in match_result.matches:
            detection = detections[det_idx]
            self.tracks[track_idx].update(
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                embedding=embeddings[det_idx],
                frame_index=self.frame_index,
                timestamp_seconds=timestamp_seconds,
            )

        reactivated_det_indices = self._reactivate_inactive_tracks(
            detections,
            embeddings,
            match_result.unmatched_detections,
            timestamp_seconds=timestamp_seconds,
        )

        for det_idx in match_result.unmatched_detections:
            if det_idx in reactivated_det_indices:
                continue
            detection = detections[det_idx]
            if self._is_duplicate_detection(detection):
                continue
            track = Track(
                track_id=0,
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                embedding=embeddings[det_idx],
                source_name=self.source_name,
                first_frame=self.frame_index,
                last_frame=self.frame_index,
                timeline_limit=self.timeline_limit,
            )
            track.record_sighting(
                frame_index=self.frame_index,
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                timestamp_seconds=timestamp_seconds,
            )
            self.tracks.append(track)

        self._promote_confirmed_tracks()
        expired_tracks = [track for track in self.tracks if track.missed > self.max_missed]
        self._mark_exited_tracks(expired_tracks)
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        self._deduplicate_live_tracks()
        return self._visible_tracks()

    def list_track_memories(self) -> List[Dict[str, object]]:
        memories = []
        for track in self.track_registry.values():
            if track.hits < self.min_hits:
                continue
            duration_frames = max(1, track.last_frame - track.first_frame + 1)
            timestamp_values = [
                item.get("timestamp_seconds")
                for item in track.timeline
                if item.get("timestamp_seconds") is not None
            ]
            visible_duration_seconds = (
                round(float(max(timestamp_values) - min(timestamp_values)), 2)
                if len(timestamp_values) >= 2
                else None
            )
            visible_duration_seconds = track.metadata.get("visible_duration_seconds", visible_duration_seconds)
            memories.append(
                {
                    "memory_id": f"{track.source_name}:{track.track_id}",
                    "track_id": track.track_id,
                    "source_name": track.source_name,
                    "first_frame": track.first_frame,
                    "last_frame": track.last_frame,
                    "duration_frames": duration_frames,
                    "visible_duration_seconds": visible_duration_seconds,
                    "hits": track.hits,
                    "best_confidence": round(float(track.best_confidence), 4),
                    "latest_bbox": list(map(int, track.bbox)),
                    "timestamps": track.metadata.get("timestamps", [])[-40:],
                    "first_seen_timestamp_seconds": track.metadata.get("first_seen_timestamp_seconds"),
                    "last_seen_timestamp_seconds": track.metadata.get("last_seen_timestamp_seconds"),
                    "visible_frame_count": track.metadata.get("visible_frame_count", track.hits),
                    "positions": track.metadata.get("positions", [])[-40:],
                    "timeline_events": track.metadata.get("timeline_events", [])[-30:],
                    "evidence": track.metadata.get("evidence", [])[-12:],
                    "crop_url": track.metadata.get("crop_url"),
                    "best_crop_url": track.metadata.get("best_crop_url"),
                    "best_crop_frame": track.metadata.get("best_crop_frame"),
                    "best_crop_timestamp_seconds": track.metadata.get("best_crop_timestamp_seconds"),
                    "evidence_url": track.metadata.get("evidence_url"),
                    "timeline": track.timeline,
                }
            )
        return sorted(memories, key=lambda item: (item["source_name"], item["track_id"]))

    def dashboard_metrics(self) -> Dict[str, object]:
        memories = self.list_track_memories()
        if not memories:
            return {
                "total_tracks": 0,
                "active_tracks": 0,
                "frames_processed": self.frame_index,
                "avg_track_duration_frames": 0.0,
                "top_tracks": [],
            }

        avg_duration = sum(item["duration_frames"] for item in memories) / len(memories)
        top_tracks = sorted(memories, key=lambda item: item["duration_frames"], reverse=True)[:5]
        return {
            "total_tracks": len(memories),
            "active_tracks": len(self.tracks),
            "frames_processed": self.frame_index,
            "avg_track_duration_frames": round(float(avg_duration), 2),
            "top_tracks": top_tracks,
        }
