# MOT + Re-ID Video Search

A full-stack demo system for person detection, multi-object tracking, person re-identification, evidence capture, and searchable video memory.

The backend uses YOLOv8 to detect people, TorchReID/OSNet embeddings to compare identities, a Hungarian matcher to keep track IDs stable, and a small semantic index for natural-language search over tracked observations. The frontend is a Next.js dashboard for uploading video, inspecting track memory, searching by query image, searching by text, and streaming frames from a live camera.

This repository is designed for inference and experimentation. It is not a training pipeline.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [Main Workflows](#main-workflows)
- [API Reference](#api-reference)
- [Generated Data](#generated-data)
- [Models and Downloads](#models-and-downloads)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Implemented Upgrades](#implemented-upgrades)
- [Scope Notes](#scope-notes)
- [Optional Future Enhancements](#optional-future-enhancements)

## What This Project Does

The system turns video frames into reusable person memories:

1. A video is uploaded or a camera frame is sent to the API.
2. YOLOv8 detects person boxes in each frame.
3. Each person crop is encoded with a Re-ID model.
4. A tracker assigns stable `track_id` values using motion, box shape, IoU, and appearance similarity.
5. Each track keeps a small Re-ID appearance gallery so image search can match against multiple views of the same person.
6. Track metadata is saved in runtime memory.
7. Crops and evidence frames are written to disk for dashboard display.
8. Periodic track observations are indexed for text search.
9. The dashboard and APIs expose video outputs, track memories, image search, text search, and simple grounded answers.
10. Completed run memories and Re-ID embedding galleries are persisted for later dashboard and image-search use.
11. Individual track memories can be exported as reviewable video clips.

The most important outputs are:

- Annotated tracking video in `data/output/`
- Person crops in `data/crops/`
- Evidence frames in `data/evidence/`
- Exported clips in `data/clips/`
- SQLite memory database in `data/mot_reid.sqlite3`
- Persisted Re-ID embeddings in `data/embeddings/`
- Track memories returned by the FastAPI routes
- Optional ChromaDB semantic index in `data/semantic_chroma/`

## System Architecture

```text
Video upload / live frame
        |
        v
FastAPI tracking routes
        |
        v
MOTReIDPipeline
        |
        +--> YOLODetector
        |       detects person bounding boxes
        |
        +--> ReIDEncoder
        |       converts person crops into appearance embeddings
        |
        +--> MultiObjectTracker
        |       assigns and maintains stable track IDs
        |
        +--> VisionMemoryEngine
        |       saves crops, evidence frames, positions, timestamps
        |
        +--> TrackPersistenceStore
        |       stores completed memories and embeddings across restarts
        |
        +--> TrackClipExporter
        |       exports per-track review clips
        |
        +--> SemanticPersonSearchIndex
        |       indexes periodic observations for text search
        |
        +--> VideoRAGAnswerer
                summarizes retrieved evidence locally or with Groq

Next.js dashboard
        |
        v
/api/[...path] proxy
        |
        v
FastAPI backend on http://127.0.0.1:8000
```

## Repository Layout

```text
app/
  main.py
    FastAPI application, CORS setup, static file mounts, health route.

  routes/tracking.py
    Public API routes for video upload, local video run, image search,
    text search, dashboard analytics, track memories, clip export,
    and live sessions.

  services/pipeline.py
    Main orchestration layer. Connects detector, Re-ID encoder, tracker,
    memory engine, persistence, clip export, semantic index, and answer layer.

  services/runtime.py
    Shared singleton pipeline used by HTTP routes and WebSocket metrics.

  services/persistence.py
    SQLite + .npy embedding persistence for processed sources and memories.

  services/clip_export.py
    Exports per-track evidence clips into data/clips/.

  services/memory_engine.py
    Saves best crops, periodic crops, evidence frames, timestamps,
    positions, and evidence-based episode summaries.

  services/semantic_search.py
    Builds text-searchable observations from tracked crops. Uses CLIP
    when cached or downloads are enabled, and falls back to explicit
    keyword overlap on generated captions when CLIP is unavailable.

  services/rag.py
    Produces a small answer from retrieved evidence. Uses Groq-compatible
    chat completions only when `GROQ_API_KEY` is set.

models/
  yolo.py
    YOLOv8 person detector wrapper.

  reid_model.py
    TorchReID OSNet encoder wrapper.

tracking/
  matcher.py
    IoU, center distance, shape cost, cosine distance, cost matrix,
    and Hungarian assignment helpers.

  tracker.py
    Multi-object tracker, track lifecycle, inactive-track reactivation,
    track registry, dashboard metrics, and memory summaries.

utils/
  visualization.py
    Draws tracked boxes and labels onto video frames.

Frontend/
  app/page.js
    Main dashboard UI.

  app/api/[...path]/route.js
    Next.js API proxy to the FastAPI backend.

tests/
  test_matcher.py
  test_tracker.py
    Unit tests for matching costs, Hungarian assignment, stable IDs,
    expiry, reactivation, and dashboard metrics.
```

Runtime data folders are intentionally separate from source code:

```text
data/input/
data/input/uploads/
data/output/
data/crops/
data/evidence/
data/clips/
data/embeddings/
data/semantic_chroma/
```

## Requirements

Recommended local environment:

- Python 3.9 or newer
- Node.js 18 or newer
- npm
- macOS, Linux, or WSL
- Enough disk space for model weights and generated video/crop outputs

Python packages are listed in `requirements.txt`. Key dependencies:

- `ultralytics`
- `opencv-python-headless`
- `torch`
- `torchvision`
- `torchreid`
- `scipy`
- `fastapi`
- `uvicorn`
- `transformers`
- `chromadb`
- `pytest`

Frontend dependencies are listed in `Frontend/package.json`:

- Next.js 16
- React 19
- React DOM 19

## Quick Start

From the project root:

```bash
cd /Users/divyanshunagar/Desktop/mot-reid-system
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

In another terminal:

```bash
cd /Users/divyanshunagar/Desktop/mot-reid-system/Frontend
npm install
npm run dev
```

Open the dashboard:

```text
http://localhost:3000
```

Check the backend:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "jobs": {},
  "model_cache": {},
  "data_dirs": {}
}
```

## Running the Backend

Start the API server from the repository root:

```bash
.venv/bin/uvicorn app.main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

FastAPI also exposes automatic docs:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

Static output routes:

```text
/outputs/<file>
/crops/<file>
/evidence/<file>
/clips/<file>
```

These map to:

```text
data/output/
data/crops/
data/evidence/
data/clips/
```

The backend also exposes live dashboard metrics over:

```text
ws://127.0.0.1:8000/ws/dashboard
```

## Running the Frontend

Start the Next.js dashboard:

```bash
cd /Users/divyanshunagar/Desktop/mot-reid-system/Frontend
npm run dev
```

The frontend runs at:

```text
http://localhost:3000
```

The frontend does not call the backend directly from browser code. It calls `/api/...`, and `Frontend/app/api/[...path]/route.js` proxies requests to the backend.

Default proxy target:

```text
http://127.0.0.1:8000
```

Override it with:

```bash
export API_BASE_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`.

## Main Workflows

### 1. Upload a Video from the Dashboard

1. Start the backend.
2. Start the frontend.
3. Open `http://localhost:3000`.
4. Select a video.
5. Choose detector model and frame stride.
6. Run the pipeline.
7. Inspect dashboard metrics, evidence images, track memories, and output video.

`frame_stride` controls how many frames are skipped:

- `1` means process every frame.
- `2` means process every second frame.
- Higher values are faster but may reduce tracking quality.

### 2. Upload a Video with curl

```bash
curl -X POST http://127.0.0.1:8000/tracking/upload \
  -F "file=@data/input/test_video.mp4" \
  -F "detector_model=yolov8n.pt" \
  -F "frame_stride=2"
```

The response now returns a background job immediately. Poll the returned `job_id` until it reaches `completed`, then read `result`.

```bash
curl http://127.0.0.1:8000/tracking/jobs/<job_id>
```

The completed job result includes:

- `source`
- `output_path`
- `output_url`
- `frames_processed`
- `sampled_frames_processed`
- `frames_with_tracks`
- `max_track_id`
- `track_memories`
- `semantic_status`
- `dashboard_metrics`

### 3. Run Tracking on a Local Backend Path

Use this when the video already exists on the backend machine:

```bash
curl -X POST http://127.0.0.1:8000/tracking/run \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "data/input/test_video.mp4",
    "output_path": "data/output/tracked_video.mp4",
    "conf_threshold": 0.50,
    "match_threshold": 0.58
  }'
```

### 4. Search by Person Image

Run a video first so the tracker has indexed track memories.

```bash
curl -X POST "http://127.0.0.1:8000/tracking/search?top_k=5" \
  -F "file=@data/input/sample.jpg"
```

The query image is encoded with the same Re-ID model. The API compares it against each track's stored appearance gallery and returns the closest track memories.

### 5. Search by Text

Run a video first so semantic observations exist.

```bash
curl -X POST http://127.0.0.1:8000/tracking/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person wearing blue clothing near center middle",
    "top_k": 5
  }'
```

With time filters:

```bash
curl -X POST http://127.0.0.1:8000/tracking/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person wearing red clothing",
    "top_k": 5,
    "start_time_seconds": 5,
    "end_time_seconds": 20
  }'
```

Text search uses CLIP when it is available. If CLIP cannot load, the API falls back to keyword overlap against generated observation captions and exposes the loader error in `semantic_status.clip_error`.

## API Reference

### Health

```http
GET /health
GET /tracking/health
```

The root health endpoint includes job, model-cache, and data-directory readiness. `GET /metrics` exposes lightweight Prometheus-style queue/cache metrics.

### Jobs

```http
GET /tracking/jobs
GET /tracking/jobs/{job_id}
POST /tracking/jobs/{job_id}/cancel
POST /tracking/jobs/{job_id}/retry
DELETE /tracking/jobs/{job_id}
```

Video upload and local video run submit background jobs. Job status values are `queued`, `running`, `cancel_requested`, `canceled`, `failed`, and `completed`.

### Model Warm-Up

```http
POST /tracking/models/warmup
Content-Type: application/json
```

Body:

```json
{
  "detector": true,
  "reid": true,
  "clip": false
}
```

CLIP warm-up is opt-in because it may download a larger model the first time.

Returns:

```json
{"status":"ok"}
```

### Run Local Video

```http
POST /tracking/run
Content-Type: application/json
```

Body:

```json
{
  "source_path": "data/input/test_video.mp4",
  "output_path": "data/output/tracked_video.mp4",
  "conf_threshold": 0.5,
  "match_threshold": 0.58
}
```

Use this route when the file already exists on disk.

### Upload Video

```http
POST /tracking/upload
Content-Type: multipart/form-data
```

Form fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | file | yes | Video file to process. |
| `detector_model` | string | no | YOLO model path/name. Example: `yolov8n.pt`. |
| `frame_stride` | integer | no | Process every Nth frame. Minimum is `1`. |

### Image Search

```http
POST /tracking/search?top_k=5
Content-Type: multipart/form-data
```

Form fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | file | yes | Query image crop/photo. |

### Text Search

```http
POST /tracking/search/text
Content-Type: application/json
```

Body:

```json
{
  "query": "person wearing blue clothing",
  "top_k": 5,
  "start_time_seconds": null,
  "end_time_seconds": null
}
```

Notes:

- `query` must be at least 2 characters.
- `top_k` must be between 1 and 20.
- Time filters are optional.

### Track Memories

```http
GET /tracking/tracks
GET /tracking/analytics/tracks
```

Returns:

```json
{
  "track_memories": []
}
```

Get one memory:

```http
GET /tracking/analytics/tracks/{memory_id}
```

Memory IDs use this pattern:

```text
source_name:track_id
```

Example:

```text
test_video:1
```

### Dashboard Metrics

```http
GET /tracking/analytics/dashboard
```

Returns overview metrics such as:

- indexed track memories
- sources processed
- semantic observations
- persisted track memories
- persisted sources
- active tracks
- frames processed
- tracker metrics
- semantic index status

### Export Track Clip

```http
POST /tracking/clips/{memory_id}
Content-Type: application/json
```

Body:

```json
{
  "padding_frames": 0
}
```

Returns:

```json
{
  "memory_id": "source:1",
  "clip_path": "data/clips/source_1_visible_segment.mp4",
  "clip_url": "/clips/source_1_visible_segment.mp4",
  "frames_exported": 120,
  "start_frame": 100,
  "end_frame": 219,
  "padding_frames": 0
}
```

Clip export creates a full-frame source-video segment from the track's first visible frame to its last visible frame, with the tracked person boxed in each frame. It does not export a person-only crop clip.

## Generated Data

The system writes runtime files during processing.

| Path | Purpose |
| --- | --- |
| `data/input/uploads/` | Uploaded source videos. |
| `data/output/` | Annotated tracking videos. |
| `data/crops/` | Person crops and best crops. |
| `data/evidence/` | Full-frame evidence images with boxes. |
| `data/clips/` | Exported per-track evidence clips. |
| `data/embeddings/` | Persisted Re-ID embeddings as `.npy` files. |
| `data/mot_reid.sqlite3` | SQLite database for sources and track memories. |
| `data/semantic_chroma/` | Persistent ChromaDB files and semantic observation metadata. |

Important behavior:

- Track registry is process-local, but completed run memories are persisted to SQLite.
- Generated image/video files remain on disk after backend restart.
- Image search can compare against persisted embeddings from completed runs.
- Semantic observation metadata is persisted to `data/semantic_chroma/semantic_observations.jsonl` and is restored on backend startup.
- CLIP-backed vector search is used when available; caption keyword search is used as a fallback when CLIP cannot load.

## Models and Downloads

### YOLO

The default detector model is:

```text
yolov8n.pt
```

The repository also includes:

```text
yolov8s.pt
```

You can choose the model from the dashboard or pass `detector_model` to `/tracking/upload`.

Tradeoff:

- `yolov8n.pt`: faster, lighter, lower accuracy.
- `yolov8s.pt`: slower, heavier, usually better detections.

### Re-ID

The default Re-ID model name is:

```text
osnet_x0_25
```

The encoder uses TorchReID and follows the `pretrained` setting passed to `ReIDEncoder`.

If you have custom weights, set `reid_weights` in `PipelineConfig` or instantiate `ReIDEncoder` with `weights_path`.

If TorchReID cannot be imported or initialized, the backend falls back to a lightweight color-histogram encoder so the demo can still run. This fallback is useful for setup resilience, but identity matching quality is lower than OSNet.

### CLIP for Text Search

The semantic search layer attempts to load:

```text
openai/clip-vit-base-patch32
```

If CLIP cannot be loaded, text search reports the loader error and falls back to keyword overlap against generated observation captions.

## Configuration

The main pipeline config lives in `app/services/pipeline.py`.

Defaults:

```python
PipelineConfig(
    detector_model="yolov8n.pt",
    reid_weights=None,
    reid_model_name="osnet_x0_25",
    conf_threshold=0.50,
    match_threshold=0.58,
    appearance_weight=0.35,
    reid_match_threshold=0.22,
    reid_max_age=900,
    max_missed=18,
    min_hits=2,
    frame_stride=1,
    semantic_interval=10,
    memory_interval=1,
    tracker_timeline_limit=180,
)
```

Useful tuning points:

| Setting | What it affects |
| --- | --- |
| `conf_threshold` | Minimum YOLO confidence for person detections. |
| `match_threshold` | Maximum matching cost for active track assignment. |
| `appearance_weight` | Balance between Re-ID appearance and spatial/motion costs. |
| `reid_match_threshold` | Appearance threshold for reactivating inactive tracks. |
| `reid_max_age` | How long inactive tracks can be reactivated. |
| `max_missed` | Frames a live track can miss before expiring. |
| `min_hits` | Detections needed before a track gets a public positive ID. |
| `frame_stride` | Video sampling rate. Higher is faster but less precise. |
| `semantic_interval` | How often semantic observations are indexed. |
| `memory_interval` | How often track memory is updated. |
| `tracker_timeline_limit` | Maximum timeline points retained per track. |

Environment variables:

| Variable | Description |
| --- | --- |
| `API_BASE_URL` | Frontend proxy target. Defaults to `http://127.0.0.1:8000`. |
| `NEXT_PUBLIC_WS_BASE` | Frontend WebSocket target. Defaults to same-origin outside local dev. |
| `GROQ_API_KEY` | Enables Groq/OpenAI-compatible answer generation. |
| `GROQ_MODEL` | Chat model name. Defaults to `llama-3.1-8b-instant`. |
| `GROQ_API_URL` | Chat completions endpoint. |
| `MOT_REID_CACHE_DIR` | Cache directory for model/library runtime files. Defaults to `/tmp/mot-reid-cache`. |

## Testing

Run unit tests from the project root:

```bash
.venv/bin/python -m pytest
```

## Docker

Run backend and frontend together:

```bash
docker compose up --build
```

The backend is exposed on `http://127.0.0.1:8000`, the frontend on `http://localhost:3000`, generated runtime data is mounted at `./data`, and model cache files are stored in a Docker volume.

Current tests cover:

- IoU calculation
- cosine distance
- center distance and shape-change costs
- Hungarian assignment
- gating unreasonable matches
- track promotion after `min_hits`
- stable IDs across small motion
- track expiry
- inactive track reactivation by appearance
- dashboard metric summaries

These tests do not run the full YOLO or Re-ID model stack. They are intentionally fast and focused on tracking logic.

## Troubleshooting

### Backend starts, but upload fails

Check:

- The backend is running from the repository root.
- `data/` folders are writable.
- The selected video file is valid.
- OpenCV can decode the video codec.
- `yolov8n.pt` or your selected detector model exists.

### Frontend says backend internal error

Check the backend terminal. The frontend proxy forwards backend errors, but the real traceback appears in the Uvicorn process.

Also verify:

```bash
curl http://127.0.0.1:8000/health
```

### Text search returns weak results

Possible causes:

- No video has been processed in this backend session.
- `semantic_interval` has not indexed enough frames yet.
- CLIP could not be loaded.
- The query asks for attributes the system does not explicitly understand.

Check the `semantic_status.clip_error` field in the API response for the loader error.

### Image search returns poor identity matches

Possible causes:

- Query image is not a clear person crop.
- Person is too small or blurry in the source video.
- Re-ID model is not using pretrained weights.
- Clothing/pose/lighting changed significantly.
- Tracks were fragmented due to occlusion or low frame rate.

For better results:

- Use a clearer query crop.
- Process with `frame_stride=1`.
- Try `yolov8s.pt`.
- Enable or provide stronger Re-ID weights.

### Live camera does not start

Browser camera access requires a secure context. Use:

```text
http://localhost:3000
```

or:

```text
http://127.0.0.1:3000
```

Also check browser camera permissions.

### First run is slow

The first run may load model weights, initialize Torch, initialize ChromaDB, and warm up inference. Later runs in the same process are usually faster.

## Implemented Upgrades

These upgrades are now part of the actual codebase.

| Upgrade | Implemented In | What Changed |
| --- | --- | --- |
| SQLite persistence | `app/services/persistence.py`, `app/services/pipeline.py` | Completed run memories and source metadata are saved to `data/mot_reid.sqlite3`. |
| Persisted embeddings | `app/services/persistence.py` | Re-ID embeddings are saved as `.npy` files under `data/embeddings/`. |
| Persisted image search | `app/services/pipeline.py` | Image search can compare against current in-memory tracks and persisted embeddings. |
| Re-ID gallery search | `tracking/tracker.py`, `app/services/persistence.py`, `app/services/pipeline.py` | Tracks keep multiple normalized appearance samples, and image search uses the best gallery match. |
| Track clip export | `app/services/clip_export.py`, `app/routes/tracking.py`, `Frontend/app/page.js` | Track memories can export full-frame visible-duration segments into `data/clips/`. |
| WebSocket dashboard | `app/main.py`, `Frontend/app/page.js` | Dashboard metrics stream from `/ws/dashboard` every 2 seconds. |
| Shared runtime pipeline | `app/services/runtime.py` | HTTP routes and WebSocket use the same pipeline object. |
| Non-blocking video runs | `app/services/runtime.py`, `app/routes/tracking.py` | Long video processing uses a fresh pipeline and swaps it into runtime after completion, so dashboard/search reads are not held behind the full run lock. |
| Persisted semantic observations | `app/services/semantic_search.py` | Semantic observation metadata is restored from `data/semantic_chroma/semantic_observations.jsonl` after restart. |
| CLIP unavailable fallback | `app/services/semantic_search.py` | When CLIP is unavailable, text search falls back to caption keyword matches and reports the loader error. |
| Re-ID fallback encoder | `models/reid_model.py` | If TorchReID is unavailable, the encoder falls back to normalized color histograms instead of crashing startup. |
| Evidence-only RAG fallback | `app/services/rag.py` | When no LLM answer is available, the API returns ranked evidence without generating a weak local answer. |
| Episode summaries | `app/services/memory_engine.py` | Summaries now include sightings, evidence count, confidence, and simple movement classification. |
| Removed unused context parameter | `app/services/pipeline.py`, `app/routes/tracking.py` | `include_context` was removed from `process_frame` calls/signature. |

## Scope Notes

- This repo does not train YOLO, Re-ID, or CLIP models.
- SQLite stores completed memories, and semantic observation metadata is restored from `data/semantic_chroma/semantic_observations.jsonl`.
- Text search prefers CLIP; if CLIP cannot load, the API falls back to caption keyword matching and exposes `semantic_status.clip_error`.
- Re-ID quality depends heavily on pretrained weights, camera angle, crop quality, lighting, and occlusion.
- Clip export creates full-frame visible-duration clips with track overlays, not stabilized cropped person-only clips.
- The tracker is intentionally lightweight and does not include Kalman filtering, zone maps, tripwires, or behavior segmentation yet.

## Optional Future Enhancements

| Feature | Impact | Effort | Target Files |
| --- | --- | --- | --- |
| Cross-camera Re-ID | High | Medium | `app/services/pipeline.py`, new identity registry service |
| Zone analytics | High | Low | New `tracking/zones.py`, `app/services/memory_engine.py`, dashboard |
| Tripwire alerts | Medium | Low | New alert service, tracking zones/lines, frontend alert panel |
| Crowd density heatmap | Medium | Medium | New heatmap service, pipeline output rendering, dashboard |
| Kalman filter tracking | Medium | Medium | `tracking/tracker.py`, optional `filterpy` dependency |
| Attribute tagging | Medium | High | `app/services/semantic_search.py`, optional classifier/pose model |
| Re-ID gallery clustering | High | Medium | New identity clustering service, persistence layer, frontend gallery |

## Good Demo Script

1. Start the backend.
2. Start the frontend.
3. Open `http://localhost:3000`.
4. Upload `data/input/test_video.mp4`.
5. Use `frame_stride=2` for speed or `frame_stride=1` for better tracking.
6. Review indexed memories and evidence images.
7. Search with `data/input/sample.jpg`.
8. Try a text query such as:

```text
person wearing blue clothing near center middle
```

9. Open the annotated video from the output URL or `data/output/`.

## Development Notes

- Keep source changes separate from generated files in `data/`.
- Prefer adding tests under `tests/` when changing matching or tracker behavior.
- The FastAPI app creates output directories on startup.
- The Next.js proxy has `maxDuration = 300`, which is useful for longer upload-processing requests.
- For reliable offline demos, cache or vendor model files before presenting.
