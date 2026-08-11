"""Track-observation scoring kept separate from storage and model inference."""
from __future__ import annotations


def combine_scores(*, visual: float | None, keyword: float, attribute: float, quality: float) -> dict[str, float]:
    """Return a bounded relevance score and its transparent components."""
    visual_score = max(0.0, float(visual or 0.0))
    keyword_score = max(0.0, min(1.0, float(keyword)))
    attribute_score = max(0.0, min(1.0, float(attribute)))
    quality_score = max(0.0, min(1.0, float(quality)))
    if visual is None:
        relevance = 0.55 * keyword_score + 0.35 * attribute_score + 0.10 * quality_score
    else:
        relevance = 0.55 * visual_score + 0.15 * keyword_score + 0.20 * attribute_score + 0.10 * quality_score
    return {
        "relevance": round(min(1.0, relevance), 4),
        "visual": round(visual_score, 4),
        "keyword": round(keyword_score, 4),
        "attributes": round(attribute_score, 4),
        "quality": round(quality_score, 4),
    }


def confidence_band(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "review"
    return "low"
