#!/usr/bin/env python3
"""Evaluate text-to-person retrieval against labelled, real-video cases.

The target video must already be processed in the supplied dashboard session.
Use ``__no_match__`` as expected_memory_id for a query whose person or
attribute combination does not exist in that video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


NO_MATCH = "__no_match__"


def search(api_url: str, session_id: str, case: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    payload = {"query": case["query"], "top_k": top_k}
    for key in ("start_time_seconds", "end_time_seconds"):
        if key in case:
            payload[key] = case[key]
    request = Request(
        f"{api_url.rstrip('/')}/tracking/search/text",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-session-id": session_id},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8")).get("matches", [])
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Search failed for {case['query']!r}: HTTP {error.code}: {detail}") from error


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        if not isinstance(record.get("query"), str) or not record["query"].strip():
            raise ValueError(f"Line {line_number} needs a non-empty query.")
        if not record.get("expected_memory_id"):
            raise ValueError(f"Line {line_number} needs expected_memory_id (or {NO_MATCH}).")
        cases.append(record)
    if not cases:
        raise ValueError("No text evaluation cases found.")
    return cases


def metrics(rankings: list[list[dict[str, Any]]], cases: list[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    top1 = top5 = reciprocal_rank = tp = fp = fn = tn = 0
    positives = negatives = 0
    for ranking, case in zip(rankings, cases):
        expected = str(case["expected_memory_id"])
        accepted = [item for item in ranking if float(item.get("score", 0.0)) >= threshold]
        accepted_ids = [str(item.get("memory_id", "")) for item in accepted]
        if expected == NO_MATCH:
            negatives += 1
            if accepted_ids:
                fp += 1
            else:
                tn += 1
            continue
        positives += 1
        if accepted_ids and accepted_ids[0] == expected:
            top1 += 1
            tp += 1
        elif accepted_ids:
            fp += 1
            fn += 1
        else:
            fn += 1
        if expected in accepted_ids[:5]:
            top5 += 1
            reciprocal_rank += 1 / (accepted_ids.index(expected) + 1)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "threshold": threshold,
        "queries": len(cases),
        "positive_queries": positives,
        "negative_queries": negatives,
        "top1": top1 / max(1, positives),
        "recall_at_5": top5 / max(1, positives),
        "mrr_at_5": reciprocal_rank / max(1, positives),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "no_match_accuracy": tn / max(1, negatives),
        "false_positive_rate": fp / max(1, fp + tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="JSONL case file; see evaluation/README.md")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("evaluation/text-search-report.json"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    rankings = [search(args.api_url, args.session_id, case, args.top_k) for case in cases]
    candidates = [metrics(rankings, cases, round(value / 100, 2)) for value in range(20, 91, 5)]
    recommended = max(candidates, key=lambda item: (item["f1"], item["no_match_accuracy"], item["recall_at_5"]))
    report = {"recommended": recommended, "all_thresholds": candidates, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(recommended, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
