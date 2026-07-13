from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(vec_a, vec_b) / denom)


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_size(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def normalized_center_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    aw, ah = box_size(box_a)
    bw, bh = box_size(box_b)
    scale = max(1.0, ((aw + bw) * 0.5 + (ah + bh) * 0.5) * 0.5)
    return float(np.hypot(ax - bx, ay - by) / scale)


def shape_change_cost(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    aw, ah = box_size(box_a)
    bw, bh = box_size(box_b)
    area_ratio = max((aw * ah) / (bw * bh), (bw * bh) / (aw * ah))
    aspect_ratio_a = aw / ah
    aspect_ratio_b = bw / bh
    aspect_ratio = max(aspect_ratio_a / aspect_ratio_b, aspect_ratio_b / aspect_ratio_a)
    return min(1.0, 0.35 * max(0.0, area_ratio - 1.0) + 0.45 * max(0.0, aspect_ratio - 1.0))


def predicted_bbox(track) -> Sequence[float]:
    if hasattr(track, "predicted_bbox"):
        return track.predicted_bbox()
    return track.bbox


@dataclass
class MatchResult:
    matches: List[Tuple[int, int]]
    unmatched_tracks: List[int]
    unmatched_detections: List[int]


def build_cost_matrix(tracks, detections, embeddings, appearance_weight: float = 0.65) -> np.ndarray:
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    costs = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for track_idx, track in enumerate(tracks):
        for det_idx, detection in enumerate(detections):
            track_box = predicted_bbox(track)
            det_box = detection["bbox"]
            iou = compute_iou(track_box, det_box)
            iou_cost = 1.0 - iou
            appearance_cost = cosine_distance(track.embedding, embeddings[det_idx])
            center_cost = min(1.0, normalized_center_distance(track_box, det_box))
            shape_cost = shape_change_cost(track_box, det_box)
            missed = getattr(track, "missed", 0)

            if center_cost > 1.35:
                costs[track_idx, det_idx] = 1e6
                continue
            if iou < 0.04 and center_cost > 0.62 and appearance_cost > 0.22:
                costs[track_idx, det_idx] = 1e6
                continue
            if missed > 4 and (center_cost > 0.85 or appearance_cost > 0.2):
                costs[track_idx, det_idx] = 1e6
                continue
            if missed > 10 and iou < 0.12:
                costs[track_idx, det_idx] = 1e6
                continue
            if shape_cost > 0.65 and iou < 0.22:
                costs[track_idx, det_idx] = 1e6
                continue

            motion_weight = 1.0 - appearance_weight
            spatial_cost = 0.68 * iou_cost + 0.24 * center_cost + 0.08 * shape_cost
            costs[track_idx, det_idx] = (
                appearance_weight * appearance_cost + motion_weight * spatial_cost
            )
    return costs


def hungarian_match(
    tracks,
    detections,
    embeddings: np.ndarray,
    max_cost: float = 0.75,
    appearance_weight: float = 0.65,
) -> MatchResult:
    if not tracks:
        return MatchResult(matches=[], unmatched_tracks=[], unmatched_detections=list(range(len(detections))))
    if not detections:
        return MatchResult(matches=[], unmatched_tracks=list(range(len(tracks))), unmatched_detections=[])

    cost_matrix = build_cost_matrix(tracks, detections, embeddings, appearance_weight=appearance_weight)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    matched_tracks = set()
    matched_detections = set()
    for row_idx, col_idx in zip(row_indices.tolist(), col_indices.tolist()):
        if cost_matrix[row_idx, col_idx] <= max_cost:
            matches.append((row_idx, col_idx))
            matched_tracks.add(row_idx)
            matched_detections.add(col_idx)

    unmatched_tracks = [idx for idx in range(len(tracks)) if idx not in matched_tracks]
    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_detections]
    return MatchResult(matches=matches, unmatched_tracks=unmatched_tracks, unmatched_detections=unmatched_detections)
