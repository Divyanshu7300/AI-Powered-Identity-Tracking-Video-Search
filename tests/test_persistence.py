from __future__ import annotations

import numpy as np

from app.services.persistence import TrackPersistenceStore


class _Track:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def embedding_candidates(self) -> np.ndarray:
        return self.embedding.reshape(1, -1)


def _memory(memory_id: str, track_id: int) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "source_name": "source-a",
        "track_id": track_id,
        "first_frame": 1,
        "last_frame": 4,
        "duration_frames": 4,
        "best_confidence": 0.9,
    }


def test_save_run_removes_tracks_missing_from_a_rerun(tmp_path) -> None:
    store = TrackPersistenceStore(
        db_path=str(tmp_path / "memories.sqlite3"),
        embeddings_dir=str(tmp_path / "embeddings"),
    )
    first = [_memory("source-a:1", 1), _memory("source-a:2", 2)]
    store.save_run(
        source_name="source-a",
        source_path="input.mp4",
        output_path="output.mp4",
        memories=first,
        track_registry={1: _Track([1, 0]), 2: _Track([0, 1])},
    )

    store.save_run(
        source_name="source-a",
        source_path="input.mp4",
        output_path="output.mp4",
        memories=[first[0]],
        track_registry={1: _Track([1, 0])},
    )

    assert [memory["memory_id"] for memory in store.list_memories()] == ["source-a:1"]
    assert not (tmp_path / "embeddings" / "source-a_2.npy").exists()
