# MOT + Re-ID Video Search & Vision RAG System

A full-stack AI system for person detection, multi-object tracking (MOT), person re-identification (Re-ID), GPU batch frame processing, dual-zone attribute recognition, evidence capture, and searchable video memory.

The backend uses **YOLOv8** for person detection, **TorchReID (OSNet)** embeddings for identity comparison, a **Hungarian Matcher** to maintain stable track IDs, **Dual-Zone Attribute Extractor** for upper/lower clothing recognition, and **CLIP + ChromaDB + Groq RAG** for natural-language text and image search over video memories. The frontend is a Next.js 16 dashboard for uploading videos, inspecting track memories, running image/text search, and exporting evidence clips.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Testing](#testing)
- [Implemented Upgrades](#implemented-upgrades)
- [Troubleshooting](#troubleshooting)

---

## Key Capabilities

1. **GPU Batch Frame Processing**: Reads video frames in chunks (default 8 frames per batch) to perform batched YOLO detection and batched Re-ID feature encoding, accelerating pipeline throughput by 3x-5x.
2. **Dual-Zone Clothing Attribute Recognition**: Automatically segments detected person crops into upper-body (shirt, jacket, top) and lower-body (trousers, pants, jeans) zones to index detailed appearance descriptions.
3. **Multi-Object Tracking (MOT)**: Combines motion displacement, IoU, center distance, bounding box shape cost, and Re-ID cosine similarity for stable tracking across occlusions.
4. **Re-ID Appearance Gallery**: Maintains a gallery of up to 8 distinct normalized embeddings per track to handle pose, lighting, and angle variations during image search.
5. **Natural Language Text Search & Vision RAG**: Searches video observations via HuggingFace CLIP embeddings and ChromaDB vector search. Optionally generates grounded LLM answers using Groq (`llama-3.1-8b-instant`).
6. **Person Image Search**: Compares an uploaded query image against live tracks and persisted embedding galleries.
7. **Per-Track Evidence Clip Export**: Exports full-frame video segments containing the visible lifetime of a track with bounding box overlays.
8. **Session Isolation & Non-blocking API**: Session IDs isolate jobs and runtime data while read-only search and analytics endpoints operate concurrently without locking video processing.

---

## System Architecture

```text
Video Upload / Local Run
        |
        v
FastAPI Tracking Routes (`app/routes/tracking.py`)
        |
        v
MOTReIDPipeline (`app/services/pipeline.py`)
        |
        +--> YOLODetector (`models/yolo.py`)
        |       GPU Batched Person Bounding Box Detection
        |
        +--> ReIDEncoder (`models/reid_model.py`)
        |       TorchReID OSNet (512-dim) Feature Extraction
        |
        +--> MultiObjectTracker (`tracking/tracker.py`)
        |       Hungarian Matcher + Inactive Track Reactivation
        |
        +--> VisionMemoryEngine (`app/services/memory_engine.py`)
        |       Saves crops, evidence frames, timestamps, and position tracking
        |
        +--> TrackPersistenceStore (`app/services/persistence.py`)
        |       SQLite database (`data/mot_reid.sqlite3`) + `.npy` embedding gallery storage
        |
        +--> SemanticPersonSearchIndex (`app/services/semantic_search.py`)
        |       Dual-zone upper/lower clothing extraction + CLIP + ChromaDB vector index
        |
        +--> VideoRAGAnswerer (`app/services/rag.py`)
        |       Evidence grounded answer generation via Groq API
        |
        +--> TrackClipExporter (`app/services/clip_export.py`)
                Exports annotated MP4 clips to `data/clips/`

Next.js 16 Dashboard (`Frontend/`)
        |
        v
Server Proxy (`Frontend/app/api/[...path]/route.js`)
        |
        v
FastAPI Backend on http://127.0.0.1:8000
```

---

## Repository Layout

```text
app/
  main.py                FastAPI entrypoint, CORS setup, static file mounts (/outputs, /crops, /evidence, /clips).
  routes/tracking.py     API endpoints: /run, /upload, /jobs, /search, /search/text, /analytics, /clips.
  services/
    pipeline.py          Pipeline orchestrator with process_frames_batch GPU batching.
    runtime.py           Session-scoped pipeline registry and job manager singleton.
    jobs.py              Thread pool background job manager with cancel/retry capabilities.
    semantic_search.py   Dual-zone clothing attribute parser, CLIP encoder, ChromaDB vector search.
    memory_engine.py     Evidence crop generator, position tracker, episode summary builder.
    persistence.py       SQLite metadata store and .npy embedding vector persistence.
    reid_index.py        Memory vector matrix and similarity search index.
    clip_export.py       Video segment exporter for track memories.
    rag.py               Groq LLM RAG answer generator over retrieved evidence.
    model_cache.py       Thread-safe singleton model cache (YOLO, Re-ID, CLIP).

models/
  yolo.py                YOLOv8 detector wrapper supporting single and batch inference.
  reid_model.py          TorchReID OSNet encoder with fallback logging.

tracking/
  tracker.py             Multi-object tracker, track lifecycle, and gallery maintenance.
  matcher.py             Cost matrix calculations (IoU, center distance, shape, cosine distance).

utils/
  image.py               Bounding box cropper utility.
  visualization.py       OpenCV video frame annotation.

Frontend/
  app/page.js            Next.js dashboard UI.
  app/api/[...path]/     Server-side proxy forwarding API requests to FastAPI backend.
  next.config.mjs        Next.js configuration (distDir: .next-build).
  Dockerfile             Production container build definition.

data/                    Runtime storage (isolated from source code):
  input/uploads/         Uploaded video files.
  output/                Annotated tracking videos.
  crops/                 Saved person bounding box images.
  evidence/              Annotated full-frame evidence screenshots.
  clips/                 Exported track video clips.
  embeddings/            Persisted Re-ID embedding matrices (.npy).
  cache/                 Downloaded model weights (YOLO, OSNet, CLIP).
  semantic_chroma/       Persistent ChromaDB vector collection.
  mot_reid.sqlite3       SQLite database storing run memories.
```

---

## Requirements

- **Python**: 3.9 or newer
- **Node.js**: 18 or 22
- **npm**: 9 or newer
- **OS**: macOS, Linux, or WSL
- **Hardware**: GPU (CUDA) recommended for fast inference, CPU supported.

Key Python dependencies (`requirements.txt`):
- `ultralytics>=8.0.0`
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `torchreid>=0.2.5`
- `tensorboard>=2.14.0`
- `fastapi>=0.110.0`, `uvicorn>=0.29.0`
- `transformers>=4.40.0`, `chromadb>=0.5.0`
- `opencv-python-headless`, `scipy`, `numpy`, `pillow`

---

## Quick Start

### 1. Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify backend health:
```bash
curl http://127.0.0.1:8000/health
```

### 2. Frontend Setup

In a second terminal:
```bash
cd Frontend
npm install
npm run dev
```

Open your browser at `http://localhost:3000`.

---

## Docker Deployment

Run both backend and frontend using Docker Compose:

```bash
docker compose up --build
```

- **Backend**: `http://localhost:8000`
- **Frontend**: `http://localhost:3000`
- **Data Volumes**: `./data` mounted locally; model weights stored in Docker volume `model-cache`.

---

## API Reference

### Health & Metrics
- `GET /health` - Overall system health, job summary, data directory status.
- `GET /tracking/health` - Session-scoped pipeline health status.
- `GET /metrics` - Prometheus format metrics (job counts, model cache states).

### Video Processing & Jobs
- `POST /tracking/upload` (Form Data: `file`, `detector_model`, `frame_stride`) - Uploads a video and starts a background tracking job. Returns `job_id`.
- `POST /tracking/run` (JSON: `source_path`, `output_path`, `conf_threshold`, `match_threshold`) - Starts tracking on a local server video path.
- `GET /tracking/jobs` - List all background jobs in current session.
- `GET /tracking/jobs/{job_id}` - Check status and progress of a specific job.
- `POST /tracking/jobs/{job_id}/cancel` - Request job cancellation.
- `POST /tracking/jobs/{job_id}/retry` - Retry a failed/completed job.
- `DELETE /tracking/jobs/{job_id}` - Delete job from history.

### Search
- `POST /tracking/search?top_k=5` (Form Data: `file`) - Search person identity by query image crop.
- `POST /tracking/search/text` (JSON: `query`, `top_k`, `start_time_seconds`, `end_time_seconds`) - Search person observations using natural language text query (Dual-zone attribute recognition + CLIP + RAG).

### Analytics & Clips
- `GET /tracking/analytics/dashboard` - Summary metrics (total tracks, active tracks, frames processed, semantic observations).
- `GET /tracking/analytics/tracks` - List track memories.
- `GET /tracking/analytics/tracks/{memory_id}` - Detailed track memory timeline and evidence.
- `POST /tracking/clips/{memory_id}` (JSON: `padding_frames`) - Export video clip for a track.

---

## Configuration & Environment Variables

Create `.env` in root and `Frontend/.env` in the `Frontend/` folder:

### Root `.env`
```env
BACKEND_PORT=8000
FRONTEND_PORT=3000
CORS_ORIGIN=http://localhost:3000
LOG_LEVEL=INFO
MOT_REID_CACHE_DIR=data/cache
MOT_REID_MAX_UPLOAD_BYTES=536870912

# Optional Groq LLM API Key for grounded Video RAG answer generation
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
```

### `Frontend/.env`
```env
API_BASE_URL=http://127.0.0.1:8000
```

---

## Testing

Run the automated test suite from the repository root:

```bash
.venv/bin/python -m pytest
```

---

## Implemented Upgrades

| Upgrade | Description | File References |
| --- | --- | --- |
| **GPU Batch Frame Processing** | Batched YOLO detection and TorchReID encoding over 8-frame chunks (3x-5x speedup). | [yolo.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/models/yolo.py), [pipeline.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/app/services/pipeline.py) |
| **Dual-Zone Attribute Recognition** | Upper-body (shirt/jacket) and lower-body (trousers/pants) attribute parser for multi-attribute text queries. | [semantic_search.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/app/services/semantic_search.py) |
| **TorchReID 512-dim Integration** | Added `tensorboard>=2.14.0` dependency to ensure full OSNet Re-ID embeddings without silent fallback. | [requirements.txt](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/requirements.txt), [reid_model.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/models/reid_model.py) |
| **Frontend Docker Alignment** | Aligned Docker runner stage to copy `.next-build` build output matching `next.config.mjs`. | [Dockerfile](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/Frontend/Dockerfile) |
| **Workspace Cache Isolation** | Defaulted `MOT_REID_CACHE_DIR` to `data/cache` for OS-level permission safety. | [yolo.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/models/yolo.py), [.env](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/.env) |
| **Non-blocking API Concurrency** | Removed coarse pipeline locks from read/search routes for smooth concurrent querying. | [tracking.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/app/routes/tracking.py) |
| **Path Security Validation** | Added strict file location and suffix checks for job retries. | [tracking.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/app/routes/tracking.py) |
| **SQLite Memory Persistence** | Persists run memories to `data/mot_reid.sqlite3` and embeddings to `data/embeddings/`. | [persistence.py](file:///Users/divyanshunagar/Desktop/ai%20video/Ai/app/services/persistence.py) |

---

## Troubleshooting

- **TorchReID Falls Back to Color Histogram**: Ensure `tensorboard` is installed (`pip install tensorboard`) and `data/cache` directory is writable.
- **Frontend Docker Start Fails**: Verify `Frontend/Dockerfile` copies `.next-build` output folder.
- **Text Search Returns No Matches**: Make sure a video has been processed in the current session so semantic observations are indexed.
- **RAG Answers Unavailable**: Check if `GROQ_API_KEY` is set in `.env`.
