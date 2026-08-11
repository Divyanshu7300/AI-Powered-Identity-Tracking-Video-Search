from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class VideoRAGAnswerer:
    """Small answer layer over already-retrieved video evidence."""

    def __init__(self, model: str | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.provider = "groq"
        self.api_key = _env_value("GROQ_API_KEY")
        self.model = model or _env_value("GROQ_MODEL", "llama-3.1-8b-instant")
        self.api_url = _env_value("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

        # requests session with retry/backoff
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504))
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def answer(self, query: str, matches: List[Dict[str, object]]) -> Dict[str, object]:
        evidence = self._build_evidence(matches)
        if not evidence:
            return {
                "answer": "I could not find matching indexed video evidence for this query yet.",
                "provider": "local",
                "evidence": [],
            }

        if self.api_key:
            generated = self._call_chat_completion(query, evidence)
            if generated:
                return {
                    "answer": generated,
                    "provider": self.provider,
                    "model": self.model,
                    "evidence": evidence,
                }

        return {
            "answer": "Answer generation is unavailable because no LLM response was produced. Returning ranked evidence only.",
            "provider": "evidence_only",
            "evidence": evidence,
        }

    def rewrite_search_query(self, query: str) -> Dict[str, object]:
        """Improve phrasing for retrieval without inventing search constraints."""
        original = str(query or "").strip()
        fallback = {
            "original_query": original, "search_query": original, "alternate_queries": [],
            "ambiguities": [], "changed": False, "provider": "local",
        }
        if not original or not self.api_key:
            return fallback
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite a video-person search query for visual retrieval. Preserve every explicit "
                        "attribute, object, colour, location, and time constraint. Never add people, objects, "
                        "colours, actions, or certainty. Return only JSON with search_query, alternate_queries "
                        "(maximum two equivalent visual phrasings), and ambiguities (maximum three unclear terms)."
                    ),
                },
                {"role": "user", "content": original},
            ],
            "temperature": 0,
            "max_tokens": 120,
            "response_format": {"type": "json_object"},
        }
        content = self._request_completion(payload)
        if not content:
            return fallback
        try:
            parsed = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
            rewritten = str(parsed.get("search_query") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            return fallback
        if not rewritten or len(rewritten) > 320:
            return fallback
        alternates = parsed.get("alternate_queries") if isinstance(parsed, dict) else []
        ambiguities = parsed.get("ambiguities") if isinstance(parsed, dict) else []
        alternate_queries = [
            str(value).strip() for value in alternates
            if isinstance(value, str) and 3 <= len(value.strip()) <= 200 and value.casefold() != rewritten.casefold()
        ][:2] if isinstance(alternates, list) else []
        ambiguity_list = [str(value).strip() for value in ambiguities if isinstance(value, str)][:3] if isinstance(ambiguities, list) else []
        return {
            "original_query": original,
            "search_query": rewritten,
            "alternate_queries": alternate_queries,
            "ambiguities": ambiguity_list,
            "changed": rewritten.casefold() != original.casefold(),
            "provider": self.provider,
            "model": self.model,
        }

    def _build_evidence(self, matches: List[Dict[str, object]]) -> List[Dict[str, object]]:
        evidence: List[Dict[str, object]] = []
        for rank, match in enumerate(matches, start=1):
            item: Dict[str, object] = {
                "rank": int(rank),
                "track_id": _safe_cast(match.get("track_id")),
                "memory_id": _safe_cast(match.get("memory_id")),
                "source_name": _safe_cast(match.get("source_name")),
                "score": _safe_float(match.get("score")),
                "description": _truncate_text(_safe_cast(match.get("caption"))),
                "timestamp_seconds": _safe_float(match.get("timestamp_seconds")),
                "best_frame_index": _safe_cast(match.get("best_frame_index")),
                "bbox": _safe_cast(match.get("bbox")),
                "duration_frames": _safe_cast(match.get("duration_frames")),
                "timeline": (match.get("timeline") or [])[-6:],
            }
            evidence.append(item)
        return evidence

    def _call_chat_completion(self, query: str, evidence: List[Dict[str, object]]) -> str | None:
        # Limit evidence size to avoid huge payloads
        max_items = 5
        trimmed = evidence[:max_items]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a video surveillance AI intelligence assistant.\n"
                        "Provide a direct, structured summary of retrieved video evidence.\n\n"
                        "OUTPUT STRUCTURE:\n"
                        "Track IDs: <number list>\n"
                        "Time Range: <seconds range>\n"
                        "- Key visual observation 1 (clothing, color, location)\n"
                        "- Key visual observation 2 (confidence or distinction)\n\n"
                        "RULES:\n"
                        "1. Maximum 2-3 bullet points.\n"
                        "2. Do NOT write long paragraphs.\n"
                        "3. Do NOT list unmatched candidates one by one."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRetrieved Evidence:\n{json.dumps(trimmed)}",
                },
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        return self._request_completion(payload)

    def _request_completion(self, payload: Dict[str, object]) -> str | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "mot-reid-system/1.0",
        }

        try:
            start = time.time()
            resp = self.session.post(self.api_url, json=payload, headers=headers, timeout=12)
            elapsed = time.time() - start
            self.logger.debug("LLM request %s %s (%.2fs) status=%s", self.api_url, self.model, elapsed, resp.status_code)
            resp.raise_for_status()
            data = resp.json()
            # Validate expected structure
            if not isinstance(data, dict):
                self.logger.error("Unexpected RAG response shape: %r", data)
                return None
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                self.logger.error("No choices in RAG response: %r", data)
                return None
            first = choices[0]
            # Provider may use different keys; try robust extraction
            content = None
            if isinstance(first, dict):
                if "message" in first and isinstance(first["message"], dict):
                    content = first["message"].get("content")
                elif "text" in first:
                    content = first.get("text")
            if not content and isinstance(data.get("result"), dict):
                content = data["result"].get("content")
            if not content:
                self.logger.error("Unable to extract content from RAG response: %r", data)
                return None
            return str(content).strip()
        except Exception as exc:
            self.logger.exception("RAG request failed: %s", exc)
            return None


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    # Try .env in working dir first, then repo root
    env_path = Path(".env")
    if not env_path.exists():
        possible = Path(__file__).resolve().parents[2] / ".env"
        if possible.exists():
            env_path = possible
        else:
            return default

    prefix = f"{name}="
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        return line.split("=", 1)[1].strip().strip('"').strip("'") or default
    return default


def _safe_cast(value: Any) -> Any:
    # Convert numpy types and common non-serializables to plain python types
    try:
        if value is None:
            return None
        # Numpy types / arrays
        import numpy as _np

        if isinstance(value, _np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [_safe_cast(v) for v in value]
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            return _safe_cast(value.tolist())
    except Exception:
        pass
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _truncate_text(text: Any, max_len: int = 300) -> str:
    s = "" if text is None else str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
