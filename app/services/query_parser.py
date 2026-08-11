"""Small deterministic parser for constraints that matter in person search.

The embedding model still receives the original prompt. This parser extracts
high-confidence constraints so a query such as "blue shirt with backpack on
the left" does not retrieve a visually similar but clearly incompatible track.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


COLORS = {"black", "white", "red", "yellow", "green", "blue", "gray"}
SYNONYMS = {
    "grey": "gray", "dark": "black", "light": "white",
    "tshirt": "shirt", "tee": "shirt", "top": "shirt", "jacket": "shirt", "hoodie": "shirt", "coat": "shirt",
    "pants": "trousers", "pant": "trousers", "jeans": "trousers", "jean": "trousers", "shorts": "trousers",
    "backpack": "bag", "handbag": "bag", "purse": "bag",
    "mobile": "phone",
}


@dataclass(frozen=True)
class SearchIntent:
    raw_query: str
    tokens: set[str]
    upper_color: str | None = None
    lower_color: str | None = None
    required_objects: set[str] = field(default_factory=set)
    horizontal_zone: str | None = None
    vertical_zone: str | None = None


def parse_person_query(query: str) -> SearchIntent:
    normalized_tokens = [
        SYNONYMS.get(token, token)
        for token in re.findall(r"[a-z0-9]+", query.lower())
    ]
    normalized = " ".join(normalized_tokens)
    upper = re.search(r"\b(" + "|".join(COLORS) + r")\s+shirt\b", normalized)
    lower = re.search(r"\b(" + "|".join(COLORS) + r")\s+trousers\b", normalized)
    required_objects = {token for token in normalized_tokens if token in {"bag", "umbrella", "phone"}}
    return SearchIntent(
        raw_query=query,
        tokens=set(normalized_tokens),
        upper_color=upper.group(1) if upper else None,
        lower_color=lower.group(1) if lower else None,
        required_objects=required_objects,
        horizontal_zone=next((value for value in ("left", "right", "center") if value in normalized_tokens), None),
        vertical_zone=next((value for value in ("top", "bottom", "middle") if value in normalized_tokens), None),
    )
