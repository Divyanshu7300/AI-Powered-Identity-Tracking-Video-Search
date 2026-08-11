from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np


EmbeddingProvider = Callable[[Dict[str, object]], np.ndarray | None]


@dataclass
class ReIDIndexStats:
    memories_indexed: int = 0
    vectors_indexed: int = 0
    embedding_dim: int | None = None
    skipped_missing: int = 0
    skipped_dimension_mismatch: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "memories_indexed": self.memories_indexed,
            "vectors_indexed": self.vectors_indexed,
            "embedding_dim": self.embedding_dim,
            "skipped_missing": self.skipped_missing,
            "skipped_dimension_mismatch": self.skipped_dimension_mismatch,
        }


class ReIDEmbeddingIndex:
    def __init__(self, persist_dir=None) -> None:
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._memory_ids: List[str] = []
        self._memories: Dict[str, Dict[str, object]] = {}
        self._stats = ReIDIndexStats()

    def rebuild(
        self,
        memories: Iterable[Dict[str, object]],
        embedding_provider: EmbeddingProvider,
    ) -> Dict[str, object]:
        rows: List[np.ndarray] = []
        memory_ids: List[str] = []
        memories_by_id: Dict[str, Dict[str, object]] = {}
        stats = ReIDIndexStats()
        expected_dim: int | None = None

        for memory in memories:
            memory_id = str(memory.get("memory_id") or "")
            if not memory_id:
                continue
            embeddings = embedding_provider(memory)
            if embeddings is None:
                stats.skipped_missing += 1
                continue

            candidates = np.asarray(embeddings, dtype=np.float32)
            if candidates.ndim == 1:
                candidates = candidates.reshape(1, -1)
            if candidates.ndim != 2 or candidates.size == 0:
                stats.skipped_missing += 1
                continue

            if expected_dim is None:
                expected_dim = int(candidates.shape[1])
                stats.embedding_dim = expected_dim
            if int(candidates.shape[1]) != expected_dim:
                stats.skipped_dimension_mismatch += 1
                continue

            normalized = self._normalize_rows(candidates)
            rows.append(normalized)
            memory_ids.extend([memory_id] * normalized.shape[0])
            memories_by_id[memory_id] = memory

        self._matrix = np.vstack(rows).astype(np.float32) if rows else np.empty((0, expected_dim or 0), dtype=np.float32)
        self._memory_ids = memory_ids
        self._memories = memories_by_id
        stats.memories_indexed = len(memories_by_id)
        stats.vectors_indexed = int(self._matrix.shape[0])
        self._stats = stats
        return self.status()

    def search(self, query: np.ndarray, top_k: int = 5) -> Dict[str, object]:
        query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if self._matrix.size == 0:
            return {
                "matches": [],
                "status": self.status(),
                "message": "Image index is empty.",
            }
        if query_vector.size != self._matrix.shape[1]:
            return {
                "matches": [],
                "status": self.status(),
                "message": "Query embedding dimension does not match the image index.",
            }

        query_vector = self._normalize_vector(query_vector)
        scores = self._matrix @ query_vector
        best_by_memory: Dict[str, float] = {}
        for index, score in enumerate(scores.tolist()):
            memory_id = self._memory_ids[index]
            if score > best_by_memory.get(memory_id, -1.0):
                best_by_memory[memory_id] = float(score)

        ranked = sorted(best_by_memory.items(), key=lambda item: item[1], reverse=True)[:top_k]
        matches = [
            {
                **self._memories[memory_id],
                "similarity": round(float(score), 4),
                "score": round(float(score), 4),
            }
            for memory_id, score in ranked
            if memory_id in self._memories
        ]
        return {
            "matches": matches,
            "status": self.status(),
            "message": "Image search used the Re-ID vector index.",
        }

    def status(self) -> Dict[str, object]:
        return self._stats.as_dict()

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms <= 1e-9] = 1.0
        return matrix / norms

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else vector
