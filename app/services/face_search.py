from __future__ import annotations

import threading
from typing import Dict, List

import cv2
import numpy as np


class FaceSearchEngine:
    """Quality-gated face embeddings for assisted track search."""

    def __init__(self) -> None:
        self._app = None
        self._lock = threading.RLock()
        self.error: str | None = None

    def extract(self, image: np.ndarray) -> List[Dict[str, object]]:
        app = self._get_app()
        if app is None:
            return []
        with self._lock:
            faces = app.get(image)
        results = []
        for face in faces:
            bbox = [int(value) for value in face.bbox]
            quality = self._quality(image, bbox, float(face.det_score))
            if quality < 0.45:
                continue
            embedding = self._normalize(np.asarray(face.embedding, dtype=np.float32))
            results.append({"bbox": bbox, "embedding": embedding, "quality": round(quality, 3)})
        return results

    def status(self) -> Dict[str, object]:
        return {"ready": self._app is not None, "error": self.error}

    def _get_app(self):
        if self._app is not None:
            return self._app
        with self._lock:
            if self._app is not None:
                return self._app
            try:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                self._app = app
            except Exception as exc:
                self.error = str(exc)
        return self._app

    @staticmethod
    def _quality(image: np.ndarray, bbox: List[int], detection_score: float) -> float:
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return 0.0
        face_size = min(x2 - x1, y2 - y1)
        if face_size < 40:
            return 0.0
        sharpness = min(1.0, float(cv2.Laplacian(face, cv2.CV_64F).var()) / 180.0)
        size_score = min(1.0, face_size / 120.0)
        return 0.45 * detection_score + 0.35 * sharpness + 0.20 * size_score

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else vector
