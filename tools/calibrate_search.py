#!/usr/bin/env python3
"""Calibrate Face / Appearance search against labelled queries from your cameras.

Input JSONL fields: query_image, expected_memory_id. Optional: query_id.
The active API session must already contain the video that owns expected_memory_id.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import requests


def search(api_url: str, session_id: str, image_path: Path, mode: str, top_k: int) -> List[Dict[str, object]]:
    with image_path.open("rb") as image:
        response = requests.post(
            f"{api_url.rstrip('/')}/tracking/search",
            headers={"x-session-id": session_id},
            data={"mode": mode, "top_k": str(top_k)},
            files={"file": (image_path.name, image, "application/octet-stream")},
            timeout=90,
        )
    response.raise_for_status()
    return response.json().get("matches", [])


def rank_hybrid(face_matches: Iterable[Dict[str, object]], appearance_matches: Iterable[Dict[str, object]], face_weight: float) -> List[Dict[str, object]]:
    candidates: Dict[str, Dict[str, float]] = defaultdict(dict)
    for item in face_matches:
        candidates[str(item["memory_id"])]["face"] = float(item.get("face_score", item.get("score", 0)))
    for item in appearance_matches:
        candidates[str(item["memory_id"])]["appearance"] = max(0.0, float(item.get("score", 0)))
    ranked = []
    for memory_id, scores in candidates.items():
        face, appearance = scores.get("face"), scores.get("appearance")
        score = face_weight * face + (1 - face_weight) * appearance if face is not None and appearance is not None else (face if face is not None else appearance)
        ranked.append({"memory_id": memory_id, "score": float(score or 0), "face": face, "appearance": appearance})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def metrics(rankings: Iterable[List[Dict[str, object]]], expected: Iterable[str], threshold: float) -> Dict[str, float]:
    pairs = list(zip(rankings, expected))
    top1 = top5 = reciprocal_rank = true_positive = false_positive = false_negative = 0
    for ranking, target in pairs:
        accepted = [match for match in ranking if match["score"] >= threshold]
        ids = [match["memory_id"] for match in accepted]
        if ids and ids[0] == target:
            top1 += 1
        if target in ids[:5]:
            top5 += 1
            reciprocal_rank += 1 / (ids.index(target) + 1)
        if ids and ids[0] == target:
            true_positive += 1
        elif ids:
            false_positive += 1
            false_negative += 1
        else:
            false_negative += 1
    count = max(1, len(pairs))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"queries": len(pairs), "top1": top1 / count, "top5": top5 / count, "mrr_at_5": reciprocal_rank / count, "precision": precision, "recall": recall, "f1": f1}


def load_cases(path: Path) -> List[Dict[str, str]]:
    cases = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("query_image") or not record.get("expected_memory_id"):
            raise ValueError(f"Line {line_number} needs query_image and expected_memory_id.")
        image = (path.parent / record["query_image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Line {line_number}: image not found: {image}")
        cases.append({"query_image": str(image), "expected_memory_id": str(record["expected_memory_id"])})
    if not cases:
        raise ValueError("No calibration cases found.")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="JSONL labels; see evaluation/README.md")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--session-id", required=True, help="Browser session ID that processed the target video")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("evaluation/report.json"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    face_rankings, appearance_rankings, expected = [], [], []
    for case in cases:
        image = Path(case["query_image"])
        face_rankings.append(search(args.api_url, args.session_id, image, "face", args.top_k))
        appearance_rankings.append(search(args.api_url, args.session_id, image, "appearance", args.top_k))
        expected.append(case["expected_memory_id"])

    candidates = []
    for weight in [round(value / 10, 1) for value in range(0, 11)]:
        rankings = [rank_hybrid(face, appearance, weight) for face, appearance in zip(face_rankings, appearance_rankings)]
        for threshold in [round(value / 100, 2) for value in range(35, 96, 5)]:
            candidates.append({"face_weight": weight, "threshold": threshold, **metrics(rankings, expected, threshold)})
    best = max(candidates, key=lambda item: (item["f1"], item["top1"], item["top5"]))
    report = {"recommended": best, "face_only": metrics(face_rankings, expected, 0.0), "appearance_only": metrics(appearance_rankings, expected, 0.0), "all_candidates": candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["recommended"], indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
