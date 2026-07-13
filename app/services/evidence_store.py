from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


class EvidenceStore:
    """Single writer for track crops and annotated evidence frames."""

    def __init__(
        self,
        crops_dir: str = "data/crops",
        evidence_dir: str = "data/evidence",
    ) -> None:
        self.crops_dir = Path(crops_dir)
        self.evidence_dir = Path(evidence_dir)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def save_track_crop(
        self,
        source_name: str,
        track_id: int,
        frame_index: int,
        crop: np.ndarray,
    ) -> str:
        filename = self._safe_filename(f"{source_name}_{track_id}_{frame_index}.jpg")
        return self._write_crop(filename, crop)

    def save_best_track_crop(
        self,
        source_name: str,
        track_id: int,
        crop: np.ndarray,
    ) -> str:
        filename = self._safe_filename(f"{source_name}_{track_id}_best.jpg")
        return self._write_crop(filename, crop)

    def save_track_evidence(
        self,
        source_name: str,
        track_id: int,
        frame_index: int,
        frame: np.ndarray,
        bbox: Sequence[float],
        timestamp: Optional[float],
    ) -> str:
        label = f"Track {track_id}"
        if timestamp is not None:
            label = f"{label} {timestamp}s"
        filename = self._safe_filename(f"{source_name}_{track_id}_{frame_index}_evidence.jpg")
        return self._write_framed_image(
            filename=filename,
            frame=frame,
            bbox=bbox,
            label=label,
            color=(25, 154, 120),
        )

    def save_observation_frame(
        self,
        observation_id: str,
        frame: np.ndarray,
        bbox: Sequence[float],
        track_id: int,
    ) -> str:
        filename = self._safe_filename(f"{observation_id}_frame.jpg")
        return self._write_framed_image(
            filename=filename,
            frame=frame,
            bbox=bbox,
            label=f"ID {track_id}",
            color=(52, 226, 197),
        )

    def _write_crop(self, filename: str, crop: np.ndarray) -> str:
        cv2.imwrite(str(self.crops_dir / filename), crop)
        return f"/crops/{filename}"

    def _write_framed_image(
        self,
        filename: str,
        frame: np.ndarray,
        bbox: Sequence[float],
        label: str,
        color: tuple[int, int, int],
    ) -> str:
        canvas = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(self.evidence_dir / filename), canvas)
        return f"/evidence/{filename}"

    @staticmethod
    def _safe_filename(name: str) -> str:
        return name.replace(":", "_").replace("/", "_")
