from __future__ import annotations

from typing import Sequence

import numpy as np


def crop_bbox(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
    if frame is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None
