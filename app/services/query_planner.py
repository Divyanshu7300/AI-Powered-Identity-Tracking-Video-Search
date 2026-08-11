"""Plans reliable strict-to-relaxed passes for person text search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from app.services.query_parser import SearchIntent, parse_person_query


@dataclass(frozen=True)
class RetrievalPass:
    name: str
    label: str
    relaxed_fields: frozenset[str]
    queries: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    intent: SearchIntent
    passes: tuple[RetrievalPass, ...]
    ambiguities: tuple[str, ...]


class PersonSearchPlanner:
    """Creates conservative fallbacks when detections or labels are imperfect."""

    def build(self, original_query: str, llm_context: Dict[str, object] | None = None) -> QueryPlan:
        intent = parse_person_query(original_query)
        primary = str((llm_context or {}).get("search_query") or original_query).strip()
        alternate_queries = self._safe_alternates(llm_context, original_query, primary)
        passes: List[RetrievalPass] = [
            RetrievalPass("exact", "Exact constraints", frozenset(), (primary,)),
        ]
        relaxed: Set[str] = set()
        if intent.horizontal_zone or intent.vertical_zone:
            relaxed.update({"horizontal_zone", "vertical_zone"})
            passes.append(RetrievalPass("without_location", "Location relaxed", frozenset(relaxed), (primary,)))
        if intent.lower_color:
            relaxed.add("lower_color")
            passes.append(RetrievalPass("without_lower_colour", "Lower-clothing colour relaxed", frozenset(relaxed), (primary,)))
        if intent.upper_color:
            relaxed.add("upper_color")
            passes.append(RetrievalPass("without_upper_colour", "Upper-clothing colour relaxed", frozenset(relaxed), (primary,)))
        if intent.required_objects:
            relaxed.add("objects")
            passes.append(RetrievalPass("without_object", "Carried-object constraint relaxed", frozenset(relaxed), (primary,)))
        all_relaxed = frozenset({"upper_color", "lower_color", "objects", "horizontal_zone", "vertical_zone"})
        passes.append(RetrievalPass("semantic", "Visual semantic fallback", all_relaxed, tuple([primary, *alternate_queries])))
        return QueryPlan(
            original_query=original_query,
            intent=intent,
            passes=tuple(passes),
            ambiguities=tuple(str(item) for item in (llm_context or {}).get("ambiguities", [])[:3]),
        )

    @staticmethod
    def _safe_alternates(context: Dict[str, object] | None, original: str, primary: str) -> List[str]:
        values = (context or {}).get("alternate_queries", [])
        if not isinstance(values, list):
            return []
        seen = {original.casefold(), primary.casefold()}
        alternatives = []
        for value in values:
            text = str(value).strip()
            if 3 <= len(text) <= 200 and text.casefold() not in seen:
                alternatives.append(text)
                seen.add(text.casefold())
            if len(alternatives) == 2:
                break
        return alternatives


def describe_relaxations(fields: Set[str] | frozenset[str]) -> List[str]:
    labels = {
        "upper_color": "upper-clothing colour",
        "lower_color": "lower-clothing colour",
        "objects": "carried object",
        "horizontal_zone": "horizontal location",
        "vertical_zone": "vertical location",
    }
    return [labels[field] for field in labels if field in fields]
