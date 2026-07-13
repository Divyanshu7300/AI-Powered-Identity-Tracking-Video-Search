from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import cv2
import numpy as np

from app.services.evidence_store import EvidenceStore
from utils.image import crop_bbox


class VisionMemoryEngine:
    """Stores practical per-track evidence for the API/dashboard."""

    def __init__(
        self,
        crops_dir: str = "data/crops",
        evidence_dir: str = "data/evidence",
        crop_interval: int = 30,
        evidence_interval: int = 45,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        self.evidence_store = evidence_store or EvidenceStore(crops_dir, evidence_dir)
        self.crop_interval = max(1, int(crop_interval))
        self.evidence_interval = max(1, int(evidence_interval))

    def update_tracks(
        self,
        frame: np.ndarray,
        tracks: Iterable,
        frame_index: int,
        timestamp_seconds: Optional[float],
    ) -> None:
        for track in tracks:
            if getattr(track, "track_id", 0) <= 0:
                continue

            metadata = track.metadata
            timestamp = round(float(timestamp_seconds), 2) if timestamp_seconds is not None else None
            cx, cy = self._center(track.bbox)

            metadata.setdefault("positions", [])
            metadata.setdefault("timestamps", [])
            metadata.setdefault("evidence", [])
            metadata.setdefault("timeline_events", [])

            if metadata.get("first_seen_timestamp_seconds") is None and timestamp is not None:
                metadata["first_seen_timestamp_seconds"] = timestamp
            if timestamp is not None:
                metadata["last_seen_timestamp_seconds"] = timestamp
                metadata["timestamps"].append({"frame_index": frame_index, "timestamp_seconds": timestamp})
                metadata["timestamps"] = metadata["timestamps"][-180:]
                first_seen = metadata.get("first_seen_timestamp_seconds")
                if first_seen is not None:
                    metadata["visible_duration_seconds"] = round(max(0.0, timestamp - float(first_seen)), 2)

            metadata["visible_frame_count"] = int(getattr(track, "hits", 0))
            metadata["positions"].append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "center": [round(cx, 2), round(cy, 2)],
                    "bbox": list(map(int, track.bbox)),
                }
            )
            metadata["positions"] = metadata["positions"][-120:]

            best_crop_url = self._save_best_crop(frame, track, frame_index, timestamp)
            crop_url = self._save_crop(frame, track, frame_index)
            evidence_url = self._save_evidence(frame, track, frame_index, timestamp)

            if crop_url:
                metadata["crop_url"] = crop_url
            if evidence_url:
                metadata["evidence_url"] = evidence_url
            if crop_url or evidence_url or best_crop_url:
                metadata["evidence"].append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "crop_url": crop_url,
                        "best_crop_url": best_crop_url,
                        "evidence_url": evidence_url,
                    }
                )
                metadata["evidence"] = metadata["evidence"][-20:]

            if not metadata["timeline_events"]:
                metadata["timeline_events"].append(
                    {
                        "type": "entered",
                        "label": "entered",
                        "frame_index": int(getattr(track, "first_frame", frame_index)),
                        "timestamp_seconds": timestamp,
                    }
                )

    def build_basic_summary(self, memory: Dict[str, object]) -> Dict[str, object]:
        duration = memory.get("visible_duration_seconds")
        if duration is None:
            duration_label = f"{memory.get('duration_frames')} frames"
        else:
            duration_label = f"{duration}s"
        evidence_count = len(memory.get("evidence") or [])
        positions = memory.get("positions") or []
        movement_label = "limited movement"
        if len(positions) >= 2:
            first = positions[0].get("center") or [0, 0]
            last = positions[-1].get("center") or [0, 0]
            distance = float(np.hypot(float(last[0]) - float(first[0]), float(last[1]) - float(first[1])))
            movement_label = "moving through scene" if distance > 120 else "mostly stationary"
        confidence = min(1.0, max(0.0, float(memory.get("best_confidence") or 0.0)))
        explanation = [
            f"{memory.get('hits', 0)} sightings",
            f"{evidence_count} saved evidence items",
            movement_label,
        ]
        return {
            "summary": (
                f"Track {memory.get('track_id')} visible for {duration_label}; "
                f"{movement_label} with {evidence_count} evidence item"
                f"{'s' if evidence_count != 1 else ''}."
            ),
            "episode_label": "tracked person",
            "activity": movement_label,
            "confidence": round(confidence, 2),
            "explanation": explanation,
        }

    def _save_crop(self, frame: np.ndarray, track, frame_index: int) -> Optional[str]:
        if frame_index % self.crop_interval != 0 and track.metadata.get("crop_url"):
            return None
        crop = crop_bbox(frame, track.bbox)
        if crop is None:
            return None
        return self.evidence_store.save_track_crop(track.source_name, track.track_id, frame_index, crop)

    def _save_best_crop(self, frame: np.ndarray, track, frame_index: int, timestamp: Optional[float]) -> Optional[str]:
        crop = crop_bbox(frame, track.bbox)
        if crop is None:
            return None
        score = self._crop_score(crop, float(getattr(track, "confidence", 0.0)))
        if score <= float(track.metadata.get("best_crop_score", 0.0)) and track.metadata.get("best_crop_url"):
            return None

        crop_url = self.evidence_store.save_best_track_crop(track.source_name, track.track_id, crop)
        track.metadata["best_crop_score"] = round(score, 4)
        track.metadata["best_crop_url"] = crop_url
        track.metadata["best_crop_frame"] = int(frame_index)
        track.metadata["best_crop_timestamp_seconds"] = timestamp
        return crop_url

    def _save_evidence(self, frame: np.ndarray, track, frame_index: int, timestamp: Optional[float]) -> Optional[str]:
        if frame_index % self.evidence_interval != 0 and track.metadata.get("evidence_url"):
            return None
        return self.evidence_store.save_track_evidence(
            track.source_name,
            track.track_id,
            frame_index,
            frame,
            track.bbox,
            timestamp,
        )

    @staticmethod
    def _center(bbox: Sequence[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _crop_score(crop: np.ndarray, confidence: float) -> float:
        if crop.shape[0] < 12 or crop.shape[1] < 8:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return confidence * 1000.0 + min(float(crop.shape[0] * crop.shape[1]) / 900.0, 450.0) + min(sharpness / 12.0, 200.0)
