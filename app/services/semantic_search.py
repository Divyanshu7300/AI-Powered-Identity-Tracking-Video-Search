from __future__ import annotations

import math
import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from app.services.evidence_store import EvidenceStore
from app.services import model_cache
from app.services.query_parser import SearchIntent, parse_person_query
from app.services.search_reranker import combine_scores, confidence_band
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
MIN_CLIP_SCORE = 0.15
MIN_KEYWORD_SCORE = 0.25


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
    source_label: str
    embedding: np.ndarray
    objects: List[str] = field(default_factory=list)
    upper_color: str = "gray"
    lower_color: str = "gray"
    has_bag: bool = False
    has_umbrella: bool = False
    has_phone: bool = False
    location: str = "center middle"
    quality_score: float = 0.0

    def metadata(self) -> Dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "track_id": self.track_id,
            "source_name": self.source_name,
            "source_label": self.source_label,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds if self.timestamp_seconds is not None else -1.0,
            "bbox": ",".join(str(value) for value in self.bbox),
            "caption": self.caption,
            "crop_url": self.crop_url,
            "frame_url": self.frame_url,
            "objects": ",".join(self.objects),
            "upper_color": self.upper_color,
            "lower_color": self.lower_color,
            "has_bag": "true" if self.has_bag else "false",
            "has_umbrella": "true" if self.has_umbrella else "false",
            "has_phone": "true" if self.has_phone else "false",
            "location": self.location,
            "quality_score": round(float(self.quality_score), 4),
        }


class SemanticPersonSearchIndex:
    """Small CLIP/Chroma/VLM-backed index over tracked person crops."""

    def __init__(
        self,
        persist_dir: str = "data/semantic_chroma",
        collection_name: str = "person_track_observations",
        crops_dir: str = "data/crops",
        evidence_dir: str = "data/evidence",
        evidence_store: EvidenceStore | None = None,
        representatives_per_track: int = 8,
        minimum_frame_gap: int = 30,
        evidence_frames_per_track: int = 3,
        semantic_model_name: str = "google/siglip-base-patch16-224",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.persist_dir / "semantic_observations.jsonl"
        self.semantic_model_name = semantic_model_name
        # Vector stores cannot mix dimensions from different embedding models.
        # Keep a separate persistent collection for each chosen model.
        model_suffix = hashlib.sha1(semantic_model_name.encode("utf-8")).hexdigest()[:10]
        self.collection_name = f"{collection_name}_{model_suffix}"
        self.evidence_store = evidence_store or EvidenceStore(crops_dir, evidence_dir)
        self.representatives_per_track = max(1, int(representatives_per_track))
        self.minimum_frame_gap = max(0, int(minimum_frame_gap))
        self.evidence_frames_per_track = max(1, int(evidence_frames_per_track))
        self.observations: Dict[str, SemanticObservation] = {}
        self._indexed_ids: set[str] = set()
        # VLM captioning is optional and can be very slow on a CPU (or pause
        # while a model downloads).  Keep normal video runs responsive; users
        # can opt into `florence`, `blip`, or `auto` in their environment.
        self.caption_model_setting = os.getenv("CAPTION_MODEL", "heuristic").lower()
        # CLIP has the same first-use download/startup cost as VLM models. It
        # is useful for richer search, but must not block an ordinary upload.
        self.enable_clip = os.getenv("SEMANTIC_ENABLE_CLIP", "false").lower() in {"1", "true", "yes", "on"}
        self._vlm_errors: Dict[str, str] = {}
        self._florence_failed = False
        self._blip_failed = False
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
            "caption_model_setting": self.caption_model_setting,
            "clip_enabled": self.enable_clip,
            "semantic_model": self.semantic_model_name,
            "clip_ready": self._clip_ready,
            "clip_error": self._clip_error,
            "vlm_errors": self._vlm_errors,
            "representatives_per_track": self.representatives_per_track,
            "evidence_frames_per_track": self.evidence_frames_per_track,
        }


    def set_clip_enabled(self, enabled: bool) -> None:
        self.enable_clip = bool(enabled)
        if not self.enable_clip:
            self._clip_ready = False
            self._clip_model = False
            self._clip_processor = None
            self._clip_lock = None
            self._clip_error = "CLIP is disabled for fast video processing."
        elif self._clip_model is False:
            self._clip_model = None
            self._clip_error = None

    def add_track_observations(
        self,
        frame: np.ndarray,
        tracks: Iterable[Dict[str, object]],
        source_name: str,
        frame_index: int,
        timestamp_seconds: Optional[float] = None,
        frame_objects: Optional[List[Dict[str, object]]] = None,
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

            quality_score = self._crop_quality(crop, track)
            if not self._should_keep_observation(source_name, track_id, frame_index, quality_score):
                continue

            associated_objects = self._associate_objects(bbox, frame_objects or [])
            (
                caption,
                upper_color,
                lower_color,
                has_bag,
                has_umbrella,
                has_phone,
                location_str,
            ) = self._generate_vlm_caption(crop, bbox, frame.shape, associated_objects)

            embedding = self._encode_image(crop)
            crop_url = self.evidence_store.save_observation_crop(observation_id, crop)
            frame_url = ""
            if self._should_save_evidence(source_name, track_id, quality_score):
                self._make_room_for_evidence(source_name, track_id, quality_score)
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
                source_label=str(track.get("source_label") or source_name),
                embedding=embedding if embedding is not None else np.empty((0,), dtype=np.float32),
                objects=associated_objects,
                upper_color=upper_color,
                lower_color=lower_color,
                has_bag=has_bag,
                has_umbrella=has_umbrella,
                has_phone=has_phone,
                location=location_str,
                quality_score=quality_score,
            )
            self._replace_weaker_nearby_observation(item)
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
            # Selection can replace an earlier, weaker representative. Rewriting
            # keeps the JSONL sidecar in sync instead of retaining stale rows.
            self._rewrite_observation_metadata()

    def _track_observations(self, source_name: str, track_id: int) -> List[SemanticObservation]:
        return [
            observation
            for observation in self.observations.values()
            if observation.source_name == source_name and observation.track_id == track_id
        ]

    def _should_keep_observation(
        self, source_name: str, track_id: int, frame_index: int, quality_score: float
    ) -> bool:
        existing = self._track_observations(source_name, track_id)
        nearby = [
            observation for observation in existing
            if abs(observation.frame_index - frame_index) <= self.minimum_frame_gap
        ]
        if nearby:
            return quality_score > max(item.quality_score for item in nearby)
        if len(existing) < self.representatives_per_track:
            return True
        return quality_score > min(item.quality_score for item in existing)

    def _replace_weaker_nearby_observation(self, candidate: SemanticObservation) -> None:
        existing = self._track_observations(candidate.source_name, candidate.track_id)
        nearby = [
            observation for observation in existing
            if abs(observation.frame_index - candidate.frame_index) <= self.minimum_frame_gap
        ]
        removable = nearby
        if not removable and len(existing) >= self.representatives_per_track:
            removable = [min(existing, key=lambda item: item.quality_score)]
        if not removable:
            return
        weakest = min(removable, key=lambda item: item.quality_score)
        if weakest.quality_score >= candidate.quality_score:
            return
        self.observations.pop(weakest.observation_id, None)
        self._indexed_ids.discard(weakest.observation_id)
        self.evidence_store.delete_evidence_url(weakest.frame_url)
        if self._collection is not None:
            try:
                self._collection.delete(ids=[weakest.observation_id])
            except Exception:
                pass

    def _should_save_evidence(self, source_name: str, track_id: int, quality_score: float) -> bool:
        evidence = [
            observation for observation in self._track_observations(source_name, track_id)
            if observation.frame_url
        ]
        return len(evidence) < self.evidence_frames_per_track or quality_score > min(
            item.quality_score for item in evidence
        )

    def _make_room_for_evidence(self, source_name: str, track_id: int, quality_score: float) -> None:
        evidence = [
            observation for observation in self._track_observations(source_name, track_id)
            if observation.frame_url
        ]
        if len(evidence) < self.evidence_frames_per_track:
            return
        weakest = min(evidence, key=lambda item: item.quality_score)
        if weakest.quality_score >= quality_score:
            return
        self.evidence_store.delete_evidence_url(weakest.frame_url)
        weakest.frame_url = ""

    def _enforce_evidence_limit(self, source_name: str) -> int:
        """Trim historical indexes to the same evidence-frame cap."""
        by_track: Dict[int, List[SemanticObservation]] = {}
        for observation in self.observations.values():
            if observation.source_name == source_name and observation.frame_url:
                by_track.setdefault(observation.track_id, []).append(observation)
        removed = 0
        for observations in by_track.values():
            observations.sort(key=lambda item: item.quality_score, reverse=True)
            for observation in observations[self.evidence_frames_per_track:]:
                self.evidence_store.delete_evidence_url(observation.frame_url)
                observation.frame_url = ""
                removed += 1
        return removed

    def _crop_quality(self, crop: np.ndarray, track: Dict[str, object]) -> float:
        """Return a bounded quality score favouring usable, sharp person crops."""
        if crop is None or crop.size == 0:
            return 0.0
        height, width = crop.shape[:2]
        area_score = min(1.0, math.sqrt((height * width) / 30_000.0))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, sharpness / 150.0)
        confidence = float(track.get("confidence", 0.5) or 0.5)
        confidence = min(1.0, max(0.0, confidence))
        return round(0.45 * area_score + 0.35 * sharpness_score + 0.20 * confidence, 4)

    def _load_crop(self, crop_url: str) -> np.ndarray | None:
        if not crop_url:
            return None
        path = self.evidence_store.crops_dir / Path(str(crop_url)).name
        if not path.is_file():
            return None
        image = cv2.imread(str(path))
        return image if image is not None and image.size else None

    def reindex_source(self, source_name: str) -> Dict[str, object]:
        """Regenerate captions and embeddings from persisted semantic crops.

        This lets a user enable CLIP or a caption model after a video has been
        processed without running detector/tracker inference again.
        """
        source_name = str(source_name)
        refreshed = 0
        skipped = 0
        vector_items: List[SemanticObservation] = []
        pruned_evidence = self._enforce_evidence_limit(source_name)
        self._load_clip()
        for observation in self.observations.values():
            if observation.source_name != source_name:
                continue
            crop = self._load_crop(observation.crop_url)
            if crop is None:
                skipped += 1
                continue
            caption, upper, lower, has_bag, has_umbrella, has_phone, _ = self._generate_vlm_caption(
                crop, observation.bbox, crop.shape, observation.objects
            )
            # The stored location is relative to the original video frame. A
            # crop has its own coordinate system, so retain that original fact.
            observation.caption = re.sub(
                r"\s+near\s+(?:left|center|right)\s+(?:top|middle|bottom)\s*$",
                f" near {observation.location}",
                caption,
            )
            observation.upper_color = upper
            observation.lower_color = lower
            observation.has_bag = has_bag
            observation.has_umbrella = has_umbrella
            observation.has_phone = has_phone
            embedding = self._encode_image(crop)
            # A temporary model-loading failure must not destroy a previously
            # persisted embedding.
            if embedding is not None:
                observation.embedding = embedding
            if observation.embedding.size:
                vector_items.append(observation)
            refreshed += 1

        if self._collection is not None and vector_items:
            try:
                self._collection.upsert(
                    ids=[item.observation_id for item in vector_items],
                    embeddings=[item.embedding.tolist() for item in vector_items],
                    documents=[item.caption for item in vector_items],
                    metadatas=[item.metadata() for item in vector_items],
                )
            except Exception:
                self._collection = None
        self._rewrite_observation_metadata()
        return {
            "source_name": source_name,
            "refreshed": refreshed,
            "skipped": skipped,
            "pruned_evidence_frames": pruned_evidence,
            "clip_ready": self._clip_ready,
            "status": self.status(),
        }

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
                    pass
        self._rewrite_observation_metadata()

    def clear_all(self) -> None:
        """Remove all indexed observations, clear vector collection, and delete JSONL sidecar."""
        self.observations.clear()
        self._indexed_ids.clear()
        if self._collection is not None:
            try:
                stored = self._collection.get(include=[])
                ids = stored.get("ids") or []
                if ids:
                    self._collection.delete(ids=ids)
            except Exception:
                pass
        if self.metadata_path.exists():
            try:
                self.metadata_path.unlink(missing_ok=True)
            except Exception:
                pass

    def search(
        self,
        query: str,
        top_k: int = 5,
        start_time_seconds: Optional[float] = None,
        end_time_seconds: Optional[float] = None,
        source_name: Optional[str] = None,
        constraints_query: Optional[str] = None,
        relaxed_fields: frozenset[str] = frozenset(),
        allow_keyword_fallback: bool = True,
    ) -> Dict[str, object]:
        self._load_clip()
        constraint_text = constraints_query or query
        query_tokens = self._tokens(query) | self._tokens(constraint_text)
        # Filters always come from the user's original words, never LLM-added text.
        intent = parse_person_query(constraint_text)
        query_embedding = self._encode_text(query) if self._clip_ready else None
        minimum_score = MIN_CLIP_SCORE if query_embedding is not None else MIN_KEYWORD_SCORE
        requested_colors = query_tokens & COLOR_WORDS
        candidates, retrieval_backend = self._retrieval_candidates(
            query_embedding=query_embedding,
            source_name=source_name,
            top_k=top_k,
        )

        scored = []
        for observation in candidates:
            if source_name and observation.source_name != source_name:
                continue
            if not self._passes_time_filter(observation, start_time_seconds, end_time_seconds):
                continue
            if requested_colors and not ({"upper_color", "lower_color"} & relaxed_fields) and not (
                requested_colors & {observation.upper_color.lower(), observation.lower_color.lower()}
            ):
                continue
            if not self._matches_requested_attributes(intent, observation, relaxed_fields):
                continue
            score, breakdown = self._hybrid_score(query_tokens, query_embedding, observation, intent)
            # Without CLIP, a low keyword/metadata overlap is not a semantic
            # match. Do not surface unrelated people merely to fill top_k.
            if score < minimum_score:
                continue
            scored.append((score, observation, breakdown))

        matches = self._group_scored_observations(scored, top_k, intent=intent)
        if not matches:
            # Approximate vector retrieval can occasionally exclude a strict
            # metadata match. Preserve recall by scanning only in that case.
            if retrieval_backend == "chromadb":
                return self._search_all_observations(
                    query=query,
                    query_tokens=query_tokens,
                    query_embedding=query_embedding,
                    intent=intent,
                    top_k=top_k,
                    start_time_seconds=start_time_seconds,
                    end_time_seconds=end_time_seconds,
                    source_name=source_name,
                    requested_colors=requested_colors,
                    relaxed_fields=relaxed_fields,
                )
            if not allow_keyword_fallback:
                return {
                    "query": query, "matches": [], "indexed_semantic_observations": len(candidates),
                    "status": self.status(), "retrieval_backend": retrieval_backend,
                    "message": "No observation matched this retrieval pass.",
                }
            return self._keyword_search(
                query=query,
                top_k=top_k,
                start_time_seconds=start_time_seconds,
                end_time_seconds=end_time_seconds,
                source_name=source_name,
                message="Semantic index active, but no observation matched query constraints.",
            )
        return {
            "query": query,
            "matches": matches,
            "indexed_semantic_observations": sum(
                1 for observation in self.observations.values()
                if not source_name or observation.source_name == source_name
            ),
            "status": self.status(),
            "retrieval_backend": retrieval_backend,
            "message": (
                f"CLIP + Metadata hybrid search active (VLM captioner: {self.caption_model_setting})."
                if self._clip_ready
                else self._clip_unavailable_message()
            ),
        }

    def _retrieval_candidates(
        self, query_embedding: np.ndarray | None, source_name: Optional[str], top_k: int
    ) -> tuple[List[SemanticObservation], str]:
        observations = [
            observation for observation in self.observations.values()
            if not source_name or observation.source_name == source_name
        ]
        # For small videos an exact in-memory scan is both faster and more
        # accurate than an approximate database query.
        if query_embedding is None or self._collection is None or len(observations) < 250:
            return observations, "memory"
        try:
            limit = min(len(observations), max(50, top_k * 15))
            query_args: Dict[str, object] = {
                "query_embeddings": [query_embedding.tolist()], "n_results": limit
            }
            if source_name:
                query_args["where"] = {"source_name": source_name}
            result = self._collection.query(**query_args)
            ids = (result.get("ids") or [[]])[0]
            candidates = [self.observations[item_id] for item_id in ids if item_id in self.observations]
            return candidates, "chromadb" if candidates else "memory"
        except Exception:
            return observations, "memory"

    def _search_all_observations(
        self,
        *,
        query: str,
        query_tokens: set[str],
        query_embedding: np.ndarray | None,
        intent: SearchIntent,
        top_k: int,
        start_time_seconds: Optional[float],
        end_time_seconds: Optional[float],
        source_name: Optional[str],
        requested_colors: set[str],
        relaxed_fields: frozenset[str],
    ) -> Dict[str, object]:
        scored = []
        for observation in self.observations.values():
            if source_name and observation.source_name != source_name:
                continue
            if not self._passes_time_filter(observation, start_time_seconds, end_time_seconds):
                continue
            if requested_colors and not ({"upper_color", "lower_color"} & relaxed_fields) and not requested_colors & {observation.upper_color.lower(), observation.lower_color.lower()}:
                continue
            if not self._matches_requested_attributes(intent, observation, relaxed_fields):
                continue
            score, breakdown = self._hybrid_score(query_tokens, query_embedding, observation, intent)
            if score >= (MIN_CLIP_SCORE if query_embedding is not None else MIN_KEYWORD_SCORE):
                scored.append((score, observation, breakdown))
        return {
            "query": query,
            "matches": self._group_scored_observations(scored, top_k, intent=intent),
            "indexed_semantic_observations": len(self.observations),
            "status": self.status(),
            "retrieval_backend": "memory_fallback",
            "message": "ChromaDB candidates were expanded with an exact metadata-safe fallback.",
        }

    def _hybrid_score(
        self,
        query_tokens: set[str],
        query_embedding: np.ndarray | None,
        observation: SemanticObservation,
        intent: SearchIntent,
    ) -> tuple[float, Dict[str, float]]:
        vector_score = 0.0
        if query_embedding is not None and observation.embedding.size > 0:
            vector_score = self._cosine(query_embedding, observation.embedding)

        keyword_score = self._keyword_score(query_tokens, observation)

        attribute_score = self._attribute_score(intent, observation)
        breakdown = combine_scores(
            visual=vector_score if query_embedding is not None and observation.embedding.size > 0 else None,
            keyword=keyword_score,
            attribute=attribute_score,
            quality=observation.quality_score,
        )
        return breakdown["relevance"], breakdown

    def _matches_requested_attributes(
        self, intent: SearchIntent, observation: SemanticObservation, relaxed_fields: frozenset[str] = frozenset()
    ) -> bool:
        """Treat explicitly requested attributes as filters, not weak hints."""
        if "upper_color" not in relaxed_fields and intent.upper_color and observation.upper_color.lower() != intent.upper_color:
            return False
        if "lower_color" not in relaxed_fields and intent.lower_color and observation.lower_color.lower() != intent.lower_color:
            return False
        if "objects" not in relaxed_fields and "umbrella" in intent.required_objects and not observation.has_umbrella:
            return False
        if "objects" not in relaxed_fields and "bag" in intent.required_objects and not observation.has_bag:
            return False
        if "objects" not in relaxed_fields and "phone" in intent.required_objects and not observation.has_phone:
            return False
        location_tokens = set(self._tokens(observation.location))
        return (
            ("horizontal_zone" in relaxed_fields or intent.horizontal_zone is None or intent.horizontal_zone in location_tokens)
            and ("vertical_zone" in relaxed_fields or intent.vertical_zone is None or intent.vertical_zone in location_tokens)
        )

    def _attribute_score(self, intent: SearchIntent, observation: SemanticObservation) -> float:
        requested = 0
        matched = 0
        for requested_value, observed_value in (
            (intent.upper_color, observation.upper_color.lower()),
            (intent.lower_color, observation.lower_color.lower()),
            (intent.horizontal_zone, set(self._tokens(observation.location))),
            (intent.vertical_zone, set(self._tokens(observation.location))),
        ):
            if requested_value is None:
                continue
            requested += 1
            if requested_value == observed_value or requested_value in observed_value:
                matched += 1
        for item in intent.required_objects:
            requested += 1
            if (item == "bag" and observation.has_bag) or (item == "umbrella" and observation.has_umbrella) or (item == "phone" and observation.has_phone):
                matched += 1
        return matched / requested if requested else 0.0


    def _load_clip(self) -> None:
        if self._clip_ready or self._clip_model is False:
            return
        if not self.enable_clip:
            self._clip_model = False
            self._clip_error = "CLIP is disabled for fast video processing. Set SEMANTIC_ENABLE_CLIP=true to enable it."
            return
        try:
            self._clip_model, self._clip_processor, self._device, self._clip_lock = model_cache.get_clip(self.semantic_model_name)
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

        objects_raw = metadata.get("objects")
        if isinstance(objects_raw, str):
            objects_list = [obj.strip() for obj in objects_raw.split(",") if obj.strip()]
        elif isinstance(objects_raw, list):
            objects_list = [str(obj) for obj in objects_raw]
        else:
            objects_list = []

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
            source_label=str(metadata.get("source_label") or source_name),
            embedding=self._embedding_array(embedding),
            objects=objects_list,
            upper_color=str(metadata.get("upper_color") or "gray"),
            lower_color=str(metadata.get("lower_color") or "gray"),
            has_bag=str(metadata.get("has_bag")).lower() == "true",
            has_umbrella=str(metadata.get("has_umbrella")).lower() == "true",
            has_phone=str(metadata.get("has_phone")).lower() == "true",
            location=str(metadata.get("location") or "center middle"),
            quality_score=self._safe_score(metadata.get("quality_score")),
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
        source_name: Optional[str],
        message: str,
    ) -> Dict[str, object]:
        query_tokens = self._tokens(query)
        scored = []
        for observation in self.observations.values():
            if source_name and observation.source_name != source_name:
                continue
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

    def _group_scored_observations(self, scored, top_k: int, intent: SearchIntent | None = None) -> List[Dict[str, object]]:
        scored.sort(key=lambda item: item[0], reverse=True)
        grouped: Dict[str, Dict[str, object]] = {}
        for entry in scored:
            score, observation = entry[:2]
            breakdown = entry[2] if len(entry) > 2 else {
                "relevance": round(float(score), 4),
                "visual": 0.0,
                "keyword": round(float(score), 4),
                "attributes": 0.0,
                "quality": round(float(observation.quality_score), 4),
            }
            current = grouped.get(observation.memory_id)
            if current is not None and current["score"] >= score:
                continue
            matched, differed = self._attribute_diff(observation, intent)
            grouped[observation.memory_id] = {
                "memory_id": observation.memory_id,
                "track_id": observation.track_id,
                "source_name": observation.source_name,
                "source_label": observation.source_label or observation.source_name,
                "score": round(float(score), 4),
                "confidence_band": confidence_band(float(score)),
                "score_breakdown": breakdown,
                "matched_attributes": matched,
                "differed_attributes": differed,
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

    def _attribute_diff(self, observation: SemanticObservation, intent: SearchIntent | None = None) -> tuple[List[str], List[str]]:
        matched: List[str] = []
        differed: List[str] = []

        if intent is None:
            return self._matched_attributes(observation), []

        # Upper color
        if intent.upper_color:
            if observation.upper_color.lower() == intent.upper_color:
                matched.append(f"{observation.upper_color} top")
            else:
                differed.append(f"Top is {observation.upper_color} (Requested {intent.upper_color})")
        else:
            matched.append(f"{observation.upper_color} top")

        # Lower color
        if intent.lower_color:
            if observation.lower_color.lower() == intent.lower_color:
                matched.append(f"{observation.lower_color} trousers")
            else:
                differed.append(f"Trousers are {observation.lower_color} (Requested {intent.lower_color})")
        else:
            matched.append(f"{observation.lower_color} trousers")

        # Required objects
        for obj in (intent.required_objects or []):
            if obj == "umbrella":
                if observation.has_umbrella:
                    matched.append("umbrella present")
                else:
                    differed.append("umbrella missing")
            elif obj == "bag":
                if observation.has_bag:
                    matched.append("bag present")
                else:
                    differed.append("bag missing")
            elif obj == "phone":
                if observation.has_phone:
                    matched.append("phone present")
                else:
                    differed.append("phone missing")

        return matched, differed

    def _matched_attributes(self, observation: SemanticObservation) -> List[str]:
        attributes = [f"{observation.upper_color} upper clothing", f"{observation.lower_color} lower clothing"]
        if observation.has_bag:
            attributes.append("bag")
        if observation.has_umbrella:
            attributes.append("umbrella")
        if observation.has_phone:
            attributes.append("phone")
        return attributes

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

    def _safe_score(self, value) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, score)) if math.isfinite(score) else 0.0

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

    def _associate_objects(self, person_bbox: List[int], frame_objects: List[Dict[str, object]]) -> List[str]:
        if not person_bbox or not frame_objects:
            return []
        px1, py1, px2, py2 = person_bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)

        margin_x = int(pw * 0.25)
        margin_y = int(ph * 0.20)
        ex1, ey1, ex2, ey2 = px1 - margin_x, py1 - margin_y, px2 + margin_x, py2 + margin_y

        associated = set()
        for obj in frame_objects:
            ob = obj.get("bbox")
            if not ob:
                continue
            ox1, oy1, ox2, oy2 = ob
            ix1 = max(ex1, ox1)
            iy1 = max(ey1, oy1)
            ix2 = min(ex2, ox2)
            iy2 = min(ey2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                name = str(obj.get("class_name") or "").lower()
                if name and name != "person":
                    associated.add(name)
        return list(associated)

    def _generate_vlm_caption(
        self,
        crop: np.ndarray,
        bbox: List[int],
        frame_shape,
        associated_objects: List[str],
    ) -> tuple[str, str, str, bool, bool, bool, str]:
        upper_color, lower_color = self._upper_lower_colors(crop)
        location_str = self._location(bbox, frame_shape)

        has_bag = any(obj in ("backpack", "handbag", "suitcase") for obj in associated_objects)
        has_umbrella = "umbrella" in associated_objects
        has_phone = any(obj in ("cell phone", "phone") for obj in associated_objects)

        vlm_caption = None

        if self.caption_model_setting not in ("heuristic", "fast"):
            # 1. Primary: Florence-2 (if setting is florence, auto, or multimodal)
            if not self._florence_failed and self.caption_model_setting in ("florence", "auto", "multimodal"):
                vlm_caption = self._generate_florence_caption(crop)

            # 2. Fallback: BLIP (if Florence-2 unavailable or blip setting)
            if not vlm_caption and not self._blip_failed and self.caption_model_setting in ("florence", "blip", "auto", "multimodal"):
                vlm_caption = self._generate_blip_caption(crop)

        # 3. Safeguard: Dual-zone color heuristic (fast, reliable)
        if not vlm_caption:
            vlm_caption = f"person wearing {upper_color} shirt and {lower_color} trousers"

        obj_str = f" with {', '.join(associated_objects)}" if associated_objects else ""
        full_caption = f"{vlm_caption}{obj_str} near {location_str}"

        return full_caption, upper_color, lower_color, has_bag, has_umbrella, has_phone, location_str

    def _generate_florence_caption(self, crop: np.ndarray) -> str | None:
        if self._florence_failed:
            return None
        try:
            model_name = os.getenv("FLORENCE_MODEL_NAME", "microsoft/Florence-2-base")
            model, processor, device, lock = model_cache.get_florence(model_name)
            image = self._to_pil(crop)
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            with lock:
                import torch

                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=256,
                        num_beams=3,
                        do_sample=False,
                    )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, task=prompt, image_size=(image.width, image.height)
            )
            caption = parsed_answer.get(prompt, "")
            if isinstance(caption, str) and len(caption.strip()) > 3:
                return caption.strip().rstrip(".")
        except Exception as exc:
            self._florence_failed = True
            self._vlm_errors["florence"] = str(exc)
        return None

    def _generate_blip_caption(self, crop: np.ndarray) -> str | None:
        if self._blip_failed:
            return None
        try:
            model_name = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
            model, processor, device, lock = model_cache.get_blip(model_name)
            image = self._to_pil(crop)
            inputs = processor(images=image, return_tensors="pt").to(device)
            with lock:
                import torch

                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=60)
            caption = processor.decode(out[0], skip_special_tokens=True)
            if isinstance(caption, str) and len(caption.strip()) > 3:
                return caption.strip().rstrip(".")
        except Exception as exc:
            self._blip_failed = True
            self._vlm_errors["blip"] = str(exc)
        return None
