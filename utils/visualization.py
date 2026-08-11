from __future__ import annotations

from typing import Iterable
import cv2
import numpy as np


# Curated palette of distinct vibrant BGR colors for multi-person track identification
PALETTE = [
    (118, 230, 0),    # Neon Emerald (BGR)
    (255, 179, 0),    # Electric Cyan
    (240, 98, 146),   # Vivid Pink
    (186, 104, 200),  # Soft Purple
    (255, 138, 101),  # Bright Coral
    (77, 208, 225),   # Turquoise
    (129, 199, 132),  # Mint Green
    (255, 213, 79),   # Warm Yellow
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a unique, vibrant BGR color for each track ID."""
    return PALETTE[int(track_id) % len(PALETTE)]


def draw_tracked_objects(
    frame: np.ndarray,
    tracks: Iterable,
    fps: float | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # Create translucent overlay for subtle box fill
    overlay = canvas.copy()
    has_fill = False

    for track in tracks:
        # Support dict or object with bbox attribute
        if isinstance(track, dict):
            bbox = track.get("bbox") or [0, 0, 0, 0]
            track_id = track.get("track_id", 0)
            confidence = float(track.get("confidence") or track.get("score") or 0.0)
        else:
            bbox = getattr(track, "bbox", [0, 0, 0, 0])
            track_id = getattr(track, "track_id", 0)
            confidence = float(getattr(track, "confidence", 0.0))

        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = _track_color(track_id)

        # 1. Subtle inner fill (8% opacity)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        has_fill = True

        # 2. Thin bounding box outline (1px)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # 3. Thick corner brackets (3px) for tech aesthetic
        corner_len = max(8, min(24, (x2 - x1) // 4, (y2 - y1) // 4))
        thickness = 3

        # Top-Left
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Top-Right
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

        # 4. Clean Label Pill above box
        pct_conf = Math_round(confidence * 100) if confidence <= 1.0 else Math_round(confidence)
        label_text = f"Subject #{track_id} | {pct_conf}%"

        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)

        label_y2 = max(th + baseline + 8, y1)
        label_y1 = max(0, label_y2 - th - baseline - 8)
        label_x2 = min(width - 1, x1 + tw + 16)

        # Dark glass label background
        cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), (20, 20, 25), -1)
        # Left accent bar
        cv2.rectangle(canvas, (x1, label_y1), (x1 + 3, label_y2), color, -1)
        # Text
        cv2.putText(
            canvas,
            label_text,
            (x1 + 8, label_y2 - baseline - 4),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            (245, 245, 250),
            1,
            cv2.LINE_AA,
        )

    # Blend translucent fill if drawn
    if has_fill:
        cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, canvas)

    # 5. Top HUD Status Banner (Live FPS)
    if fps is not None and fps > 0:
        fps_text = f"LIVE STREAM  |  {fps:.1f} FPS"
        (ftw, fth), fbase = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.rectangle(canvas, (10, 10), (10 + ftw + 24, 10 + fth + 14), (15, 15, 20), -1)
        cv2.circle(canvas, (20, 10 + (fth + 14) // 2), 4, (0, 230, 118), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            fps_text,
            (30, 10 + fth + 3),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (220, 220, 230),
            1,
            cv2.LINE_AA,
        )

    return canvas


def Math_round(val: float) -> int:
    return int(round(val))
