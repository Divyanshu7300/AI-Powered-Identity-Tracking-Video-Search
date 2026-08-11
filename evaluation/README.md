# Search calibration

Create `cases.jsonl` beside the query images. One line per known identity query:

```json
{"query_image":"queries/person_01_front.jpg","expected_memory_id":"current-video-id:7"}
```

Use at least 50 examples from the same cameras you will deploy: daylight, night, side profile, blur, occlusion, and changes of clothing. Include negative examples by setting `expected_memory_id` to a value that cannot occur in the current video (for example `__no_match__`).

Process the target video in the dashboard, then get its browser session ID from local storage key `mot-reid-session-id`. Run:

```bash
python3 tools/calibrate_search.py evaluation/cases.jsonl --session-id YOUR_SESSION_ID
```

The report provides Top-1, Top-5, MRR@5, precision, recall, F1, and a recommended `face_weight` and acceptance threshold. Adopt a threshold only after checking false matches manually; the tool supports review, not automatic identification.

## Text search calibration

Copy `text_cases.example.jsonl` to `text_cases.jsonl`. Each line contains the
natural-language query and its expected track memory. For queries where no
person should be returned, use `__no_match__`.

```json
{"query":"person with backpack near left","expected_memory_id":"current-video-id:3"}
{"query":"person with umbrella","expected_memory_id":"__no_match__"}
```

Use at least 50 cases from the same camera setup. Include both positives and
negatives, and cover daylight, low light, crowded frames, similar clothing,
blur, occlusion, and queries with a time window. Process the target video in
the dashboard, then run:

```bash
python3 tools/evaluate_text_search.py evaluation/text_cases.jsonl --session-id YOUR_SESSION_ID
```

The report contains Recall@1, Recall@5, MRR@5, precision, recall, F1,
no-match accuracy, and false-positive rate for several relevance thresholds.
Treat the reported threshold as a starting point for review, not an identity
decision threshold.
