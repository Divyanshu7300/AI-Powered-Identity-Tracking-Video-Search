import numpy as np

from tracking.matcher import build_cost_matrix, hungarian_match
from tracking.tracker import MultiObjectTracker, Track


def _track(track_id, bbox, embedding):
    return Track(track_id=track_id, bbox=bbox, confidence=0.9, embedding=np.asarray(embedding, dtype=np.float32))


def test_crossing_people_match_by_identity_not_nearest_box():
    """Crossing boxes are spatially ambiguous, so Re-ID must decide the match."""
    first = _track(1, [0, 0, 20, 40], [1.0, 0.0])
    second = _track(2, [24, 0, 44, 40], [0.0, 1.0])
    detections = [
        {"bbox": [23, 0, 43, 40], "confidence": 0.9},  # first person, now near second
        {"bbox": [1, 0, 21, 40], "confidence": 0.9},   # second person, now near first
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = hungarian_match([first, second], detections, embeddings, max_cost=0.75, appearance_weight=0.65)

    assert sorted(result.matches) == [(0, 0), (1, 1)]


def test_gallery_recovers_identity_when_latest_embedding_is_contaminated():
    track = _track(1, [0, 0, 20, 40], [1.0, 0.0])
    # Simulate an overlap frame whose crop was dominated by another person.
    track.embedding = np.asarray([0.0, 1.0], dtype=np.float32)
    detection = {"bbox": [1, 0, 21, 40], "confidence": 0.9}

    cost = build_cost_matrix([track], [detection], np.asarray([[1.0, 0.0]], dtype=np.float32), appearance_weight=0.65)

    assert cost[0, 0] < 0.1


def test_reactivates_original_id_after_an_occlusion():
    tracker = MultiObjectTracker(min_hits=1, max_missed=1, reid_match_threshold=0.32)
    tracker.update(
        [{"bbox": [0, 0, 20, 40], "confidence": 0.9}],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_index=1,
    )
    tracker.update([], np.empty((0, 2), dtype=np.float32), frame_index=2)
    tracker.update([], np.empty((0, 2), dtype=np.float32), frame_index=3)

    visible = tracker.update(
        [{"bbox": [2, 0, 22, 40], "confidence": 0.9}],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_index=4,
    )

    assert [track.track_id for track in visible] == [1]
    assert tracker.next_track_id == 2
