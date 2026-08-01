from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from app.services.evidence_store import EvidenceStore
from app.services import model_cache
from utils.image import crop_bbox


COLORS = {
    "black": np.array([25, 25, 25]),
    "white": np.array([235, 235, 235]),
    "red": np.array([210, 55, 55]),
    "yellow": np.array([220, 200, 65]),
    "green": np.array([65, 160, 95]),
    "blue": np.array([65, 110, 205]),
    "gray": np.array([130, 130, 130]),
}

TEXT_SYNONYMS = {
    "centre": "center",
    "middle": "center",
    "tshirt": "shirt",
    "tee": "shirt",
    "top": "shirt",
    "jacket": "shirt",
    "hoodie": "shirt",
    "coat": "shirt",
    "pants": "trousers",
    "pant": "trousers",
    "jeans": "trousers",
    "jean": "trousers",
    "shorts": "trousers",
    "grey": "gray",
    "dark": "black",
    "light": "white",
    "backpack": "bag",
    "handbag": "bag",
    "purse": "bag",
}


COLOR_WORDS = set(COLORS)
LOCATION_WORDS = {"left", "right", "center", "top", "bottom", "middle"}
MIN_CLIP_SCORE = 0.18
MIN_KEYWORD_SCORE = 0.32


@dataclass
class SemanticObservation:
    observation_id: str
    memory_id: str
    track_id: int
    source_name: str
    frame_index: int
    timestamp_seconds: Optional[float]
    bbox: List[int]
    caption: str
    crop_url: str
    frame_url: str
    embedding: np.ndarray

    def metadata(self) -> Dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "track_id": self.track_id,
            "source_name": self.source_name,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds if self.timestamp_seconds is not None else -1.0,
            "bbox": ",".join(str(value) for value in self.bbox),
            "caption": self.caption,
            "crop_url": self.crop_url,
            "frame_url": self.frame_url,
        }


class SemanticPersonSearchIndex:
    """Small CLIP/Chroma-backed index over tracked person crops."""

    def __init__(
        self,
        persist_dir: str = "data/semantic_chroma",
        collection_name: str = "person_track_observations",
        crops_dir: str = "data/crops",
        evidence_dir: str = "data/evidence",
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.persist_dir / "semantic_observations.jsonl"
        self.collection_name = collection_name
        self.evidence_store = evidence_store or EvidenceStore(crops_dir, evidence_dir)
        self.observations: Dict[str, SemanticObservation] = {}
        self._indexed_ids: set[str] = set()
        self._clip_model = None
        self._clip_processor = None
        self._clip_lock = None
        self._clip_ready = False
        self._clip_error: str | None = None
        self._device = "cpu"
        self._collection = self._init_chroma()
        self._restore_observations()

    def status(self) -> Dict[str, object]:
        return {
            "semantic_observations": len(self.observations),
            "vector_store": "chromadb" if self._collection is not None else "memory",
            "clip_ready": self._clip_ready,
            "clip_error": self._clip_error,
        }

    def add_track_observations(
        self,
        frame: np.ndarray,
        tracks: Iterable[Dict[str, object]],
        source_name: str,
        frame_index: int,
        timestamp_seconds: Optional[float] = None,
    ) -> None:
        if frame is None:
            return

        new_items: List[SemanticObservation] = []
        for track in tracks:
            track_id = int(track["track_id"])
            bbox = list(map(int, track["bbox"]))
            observation_id = f"{source_name}:{track_id}:{frame_index}"
            if observation_id in self._indexed_ids:
                continue

            crop = crop_bbox(frame, bbox)
            if crop is None:
                continue

            caption = self._evidence_text(crop, bbox, frame.shape)
            embedding = self._encode_image(crop)
            crop_url = str(track.get("best_crop_url") or track.get("crop_url") or "")
            frame_url = self._save_framed_image(observation_id, frame, bbox, track_id)
            item = SemanticObservation(
                observation_id=observation_id,
                memory_id=f"{source_name}:{track_id}",
                track_id=track_id,
                source_name=source_name,
                frame_index=int(frame_index),
                timestamp_seconds=timestamp_seconds,
                bbox=bbox,
                caption=caption,
                crop_url=crop_url,
                frame_url=frame_url,
                embedding=embedding if embedding is not None else np.empty((0,), dtype=np.float32),
            )
            self.observations[observation_id] = item
            self._indexed_ids.add(observation_id)
            new_items.append(item)

        if self._collection is not None and new_items:
            try:
                vector_items = [item for item in new_items if item.embedding.size > 0]
                if vector_items:
                    self._collection.add(
                        ids=[item.observation_id for item in vector_items],
                        embeddings=[item.embedding.tolist() for item in vector_items],
                        documents=[item.caption for item in vector_items],
                        metadatas=[item.metadata() for item in vector_items],
                    )
            except Exception:
                self._collection = None
        if new_items:
            self._append_observation_metadata(new_items)

    def clear_source(self, source_name: str) -> None:
        """Remove prior observations for a source before it is reprocessed."""
        source_name = str(source_name)
        stale_ids = [
            observation_id
            for observation_id, observation in self.observations.items()
            if observation.source_name == source_name
        ]
        for observation_id in stale_ids:
            self.observations.pop(observation_id, None)
            self._indexed_ids.discard(observation_id)

        if self._collection is not None:
            try:
                self._collection.delete(where={"source_name": source_name})
            except Exception:
                try:
                    if stale_ids:
                        self._collection.delete(ids=stale_ids)
                except Exception:
                    # The in-memory and metadata stores remain consistent even
                    # when an optional Chroma backend cannot be updated.
                    pass
        self._rewrite_observation_metadata()

    def search(
        self,
        query: str,
        top_k: int = 5,
        start_time_seconds: Optional[float] = None,
        end_time_seconds: Optional[float] = None,
    ) -> Dict[str, object]:
        self._load_clip()
        query_embedding = self._encode_text(query) if self._clip_ready else None
        if not self._clip_ready or query_embedding is None:
            return self._keyword_search(
                query=query,
                top_k=top_k,
                start_time_seconds=start_time_seconds,
                end_time_seconds=end_time_seconds,
                message=f"{self._clip_unavailable_message()} Falling back to caption keyword search.",
            )
        scored = []
        for observation in self.observations.values():
            if not self._passes_time_filter(observation, start_time_seconds, end_time_seconds):
                continue
            if observation.embedding.size == 0:
                continue
            score = self._cosine(query_embedding, observation.embedding)
            if score < MIN_CLIP_SCORE:
                continue
            scored.append((score, observation))

        matches = self._group_scored_observations(scored, top_k)
        if not matches:
            return self._keyword_search(
                query=query,
                top_k=top_k,
                start_time_seconds=start_time_seconds,
                end_time_seconds=end_time_seconds,
                message="CLIP semantic search is active, but no vector match was found. Falling back to caption keyword search.",
            )
        return {
            "query": query,
            "matches": matches,
            "indexed_semantic_observations": len(self.observations),
            "status": self.status(),
            "message": (
                "CLIP semantic search is active."
                if self._clip_ready
                else self._clip_unavailable_message()
            ),
        }

    def _load_clip(self) -> None:
        if self._clip_ready or self._clip_model is False:
            return
        try:
            self._clip_model, self._clip_processor, self._device, self._clip_lock = model_cache.get_clip()
            self._clip_ready = True
            self._clip_error = None
        except Exception as exc:
            self._clip_model = False
            self._clip_processor = None
            self._clip_lock = None
            self._clip_ready = False
            self._clip_error = str(exc)

    def _encode_image(self, crop: np.ndarray) -> np.ndarray | None:
        self._load_clip()
        if not self._clip_ready:
            return None
        try:
            import torch

            image = self._to_pil(crop)
            inputs = self._clip_processor(images=image, return_tensors="pt").to(self._device)
            with self._clip_lock:
                with torch.inference_mode():
                    features = self._clip_model.get_image_features(**inputs)
            return self._normalize(features.detach().cpu().numpy().reshape(-1))
        except Exception as exc:
            self._clip_error = str(exc)
            return None

    def _encode_text(self, text: str) -> np.ndarray | None:
        self._load_clip()
        if not self._clip_ready:
            return None
        try:
            import torch

            inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True).to(self._device)
            with self._clip_lock:
                with torch.inference_mode():
                    features = self._clip_model.get_text_features(**inputs)
            return self._normalize(features.detach().cpu().numpy().reshape(-1))
        except Exception as exc:
            self._clip_error = str(exc)
            return None

    def _init_chroma(self):
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_dir))
            return client.get_or_create_collection(name=self.collection_name)
        except Exception:
            return None

    def _restore_observations(self) -> None:
        self._restore_observations_from_metadata_file()
        self._restore_observations_from_chroma()

    def _restore_observations_from_metadata_file(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            for line in self.metadata_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                self._upsert_observation_from_metadata(json.loads(line), embedding=None)
        except Exception:
            return

    def _restore_observations_from_chroma(self) -> None:
        if self._collection is None:
            return
        try:
            stored = self._collection.get(include=["embeddings", "documents", "metadatas"])
        except Exception:
            return

        ids = stored.get("ids") or []
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        embeddings = stored.get("embeddings")
        if embeddings is None:
            embeddings = []
        for index, observation_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            if index < len(documents) and documents[index]:
                metadata["caption"] = documents[index]
            embedding = embeddings[index] if index < len(embeddings) else None
            self._upsert_observation_from_metadata(
                {**metadata, "observation_id": observation_id},
                embedding=embedding,
            )

    def _upsert_observation_from_metadata(self, metadata: Dict[str, object], embedding) -> None:
        observation_id = str(metadata.get("observation_id") or "")
        if not observation_id:
            return
        memory_id = str(metadata.get("memory_id") or "")
        if not memory_id:
            return
        track_id = self._safe_int(metadata.get("track_id"))
        source_name = str(metadata.get("source_name") or memory_id.split(":", 1)[0])
        frame_index = self._safe_int(metadata.get("frame_index"))
        timestamp_seconds = self._safe_optional_float(metadata.get("timestamp_seconds"))
        bbox = self._parse_bbox(metadata.get("bbox"))
        item = SemanticObservation(
            observation_id=observation_id,
            memory_id=memory_id,
            track_id=track_id,
            source_name=source_name,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            bbox=bbox,
            caption=str(metadata.get("caption") or ""),
            crop_url=str(metadata.get("crop_url") or ""),
            frame_url=str(metadata.get("frame_url") or ""),
            embedding=self._embedding_array(embedding),
        )
        existing = self.observations.get(observation_id)
        if existing is not None and existing.embedding.size > 0 and item.embedding.size == 0:
            item.embedding = existing.embedding
        self.observations[observation_id] = item
        self._indexed_ids.add(observation_id)

    def _append_observation_metadata(self, items: List[SemanticObservation]) -> None:
        try:
            with self.metadata_path.open("a", encoding="utf-8") as file:
                for item in items:
                    payload = {"observation_id": item.observation_id, **item.metadata()}
                    file.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            return

    def _rewrite_observation_metadata(self) -> None:
        try:
            with self.metadata_path.open("w", encoding="utf-8") as file:
                for item in self.observations.values():
                    payload = {"observation_id": item.observation_id, **item.metadata()}
                    file.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            return

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        start_time_seconds: Optional[float],
        end_time_seconds: Optional[float],
        message: str,
    ) -> Dict[str, object]:
        query_tokens = self._tokens(query)
        scored = []
        for observation in self.observations.values():
            if not self._passes_time_filter(observation, start_time_seconds, end_time_seconds):
                continue
            score = self._keyword_score(query_tokens, observation)
            if score < MIN_KEYWORD_SCORE:
                continue
            scored.append((score, observation))
        return {
            "query": query,
            "matches": self._group_scored_observations(scored, top_k),
            "indexed_semantic_observations": len(self.observations),
            "status": self.status(),
            "message": message,
        }

    def _group_scored_observations(self, scored, top_k: int) -> List[Dict[str, object]]:
        scored.sort(key=lambda item: item[0], reverse=True)
        grouped: Dict[str, Dict[str, object]] = {}
        for score, observation in scored:
            current = grouped.get(observation.memory_id)
            if current is not None and current["score"] >= score:
                continue
            grouped[observation.memory_id] = {
                "memory_id": observation.memory_id,
                "track_id": observation.track_id,
                "source_name": observation.source_name,
                "score": round(float(score), 4),
                "caption": observation.caption,
                "crop_url": observation.crop_url,
                "frame_url": observation.frame_url,
                "best_frame_index": observation.frame_index,
                "timestamp_seconds": (
                    round(float(observation.timestamp_seconds), 2)
                    if observation.timestamp_seconds is not None and math.isfinite(observation.timestamp_seconds)
                    else None
                ),
                "bbox": observation.bbox,
            }
        return sorted(grouped.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    def _keyword_score(self, query_tokens: set[str], observation: SemanticObservation) -> float:
        if not query_tokens:
            return 0.0
        caption_tokens = self._tokens(observation.caption)
        if not caption_tokens:
            return 0.0
        overlap = query_tokens & caption_tokens
        if not overlap:
            return 0.0
        coverage = len(overlap) / max(1, len(query_tokens))
        specificity = len(overlap) / max(1, len(caption_tokens))
        important_query_tokens = query_tokens & (COLOR_WORDS | LOCATION_WORDS)
        important_overlap = overlap & important_query_tokens
        important_bonus = 0.25 * (len(important_overlap) / max(1, len(important_query_tokens)))
        return min(1.0, 0.65 * coverage + 0.25 * specificity + important_bonus)

    def _tokens(self, text: str) -> set[str]:
        stopwords = {"a", "an", "and", "at", "by", "for", "in", "near", "of", "on", "person", "the", "to", "wearing", "clothing", "clothes"}
        return {
            TEXT_SYNONYMS.get(token, token)
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in stopwords and len(token) > 1
        }

    def _parse_bbox(self, value) -> List[int]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return [self._safe_int(part) for part in parts[:4]]
        if isinstance(value, (list, tuple)):
            return [self._safe_int(part) for part in value[:4]]
        return []

    def _embedding_array(self, embedding) -> np.ndarray:
        if embedding is None:
            return np.empty((0,), dtype=np.float32)
        array = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return self._normalize(array) if array.size else array

    def _safe_int(self, value) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _safe_optional_float(self, value) -> Optional[float]:
        try:
            number = float(value)
        except Exception:
            return None
        return number if math.isfinite(number) and number >= 0 else None

    def _evidence_text(self, crop: np.ndarray, bbox: List[int], frame_shape) -> str:
        upper_color, lower_color = self._upper_lower_colors(crop)
        location_str = self._location(bbox, frame_shape)
        return f"person wearing {upper_color} shirt and {lower_color} trousers near {location_str}"

    def _upper_lower_colors(self, crop: np.ndarray) -> tuple[str, str]:
        if crop is None or crop.size == 0:
            return "gray", "gray"
        h = crop.shape[0]
        split = max(1, h // 2)
        upper_half = crop[:split, :]
        lower_half = crop[split:, :]
        return self._dominant_color(upper_half), self._dominant_color(lower_half)

    def _dominant_color(self, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return "gray"
        resized = cv2.resize(crop, (32, 32))
        mean_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).reshape(-1, 3).mean(axis=0)
        return min(COLORS, key=lambda name: float(np.linalg.norm(mean_rgb - COLORS[name])))


    def _location(self, bbox: List[int], frame_shape) -> str:
        x1, y1, x2, y2 = map(float, bbox)
        height, width = frame_shape[:2]
        cx = ((x1 + x2) / 2.0) / max(width, 1)
        cy = ((y1 + y2) / 2.0) / max(height, 1)
        horizontal = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
        vertical = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
        return f"{horizontal} {vertical}"

    def _save_framed_image(self, observation_id: str, frame: np.ndarray, bbox: List[int], track_id: int) -> str:
        return self.evidence_store.save_observation_frame(observation_id, frame, bbox, track_id)

    def _passes_time_filter(
        self,
        observation: SemanticObservation,
        start_time_seconds: Optional[float],
        end_time_seconds: Optional[float],
    ) -> bool:
        if observation.timestamp_seconds is None:
            return True
        if start_time_seconds is not None and observation.timestamp_seconds < start_time_seconds:
            return False
        if end_time_seconds is not None and observation.timestamp_seconds > end_time_seconds:
            return False
        return True

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        left = self._normalize(left)
        right = self._normalize(right)
        if left.shape != right.shape:
            return 0.0
        return float(np.dot(left, right))

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def _to_pil(self, image: np.ndarray):
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _clip_unavailable_message(self) -> str:
        if self._clip_error:
            return f"CLIP semantic search is unavailable: {self._clip_error}"
        return "CLIP semantic search is unavailable."
