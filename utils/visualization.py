from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


BOX_THICKNESS = 2
FONT_SCALE = 0.55
FONT_THICKNESS = 2
FPS_COLOR = (0, 255, 0)


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        50 + (track_id * 47) % 205,
        120,
        255 - (track_id * 31) % 205,
    )


def draw_tracked_objects(
    frame: np.ndarray,
    tracks: Iterable,
    fps: float | None = None,
) -> np.ndarray:
    canvas = frame.copy()

    height, width = canvas.shape[:2]

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        color = _track_color(track.track_id)

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            BOX_THICKNESS,
        )

        label = f"ID {track.track_id} | {track.confidence:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            FONT_THICKNESS,
        )

        cv2.rectangle(
            canvas,
            (x1, y1 - text_height - baseline - 6),
            (x1 + text_width + 6, y1),
            color,
            -1,
        )

        cv2.putText(
            canvas,
            label,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )

    if fps is not None:
        cv2.putText(
            canvas,
            f"FPS: {fps:.2f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            FPS_COLOR,
            2,
        )

    return canvas
