from __future__ import annotations

import logging
import os
import re
import secrets
from pathlib import Path

from dotenv import load_dotenv

# Runtime creates the session pipeline at import time, so configuration must be
# loaded before importing routes/runtime.
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from app.routes.tracking import router as tracking_router
from app.services import model_cache, runtime
from app.services.auth import LoginRequest, SignupRequest, current_identity, login, signup


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Vision RAG",
    version="1.0.0",
    description="YOLO + ReID + CLIP + Video RAG system",
)

allowed_origins = [value.strip() for value in os.getenv("CORS_ORIGIN", "http://localhost:3000").split(",") if value.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = Path("data/users")
DATA_ROOT.mkdir(parents=True, exist_ok=True)


@app.post("/auth/login")
def auth_login(payload: LoginRequest, response: Response, request: Request):
    result = login(payload, client_key=request.client.host if request.client else "unknown")
    response.set_cookie("mot_reid_access_token", result["access_token"], httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE", "false").lower() == "true", max_age=result["expires_in"])
    return result


@app.post("/auth/signup", status_code=201)
def auth_signup(payload: SignupRequest, response: Response, request: Request):
    result = signup(payload, client_key=request.client.host if request.client else "unknown")
    response.set_cookie("mot_reid_access_token", result["access_token"], httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE", "false").lower() == "true", max_age=result["expires_in"])
    return result


@app.post("/auth/logout", status_code=204)
def auth_logout(response: Response):
    response.delete_cookie("mot_reid_access_token")


def _is_hash_or_uuid(val: str) -> bool:
    if not val:
        return True
    if re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", val):
        return True
    if re.match(r"^(user_|usr_|session_|anon_|[0-9a-fA-F]{24,})", val):
        return True
    return False


@app.get("/auth/me")
def auth_me(identity: dict = Depends(current_identity)):
    sub = identity.get("sub", "")
    if _is_hash_or_uuid(sub):
        return {"username": "Operator"}
    return {"username": sub}


@app.get("/media/{media_type}/{filename:path}")
def private_media(media_type: str, filename: str, request: Request):
    roots = {"outputs": "output", "crops": "crops", "evidence": "evidence", "clips": "clips"}
    if media_type not in roots:
        raise HTTPException(status_code=404, detail="Media not found.")
    
    sid = "default"
    try:
        identity = current_identity(request)
        sid = identity.get("sid", "default")
    except Exception:
        pass

    target_subfolder = roots[media_type]
    user_root = (DATA_ROOT / runtime.normalize_session_id(sid) / target_subfolder).resolve()
    path = (user_root / filename).resolve()

    if not path.is_file() and DATA_ROOT.exists():
        for user_dir in DATA_ROOT.iterdir():
            if user_dir.is_dir():
                candidate = (user_dir / target_subfolder / filename).resolve()
                if candidate.is_file():
                    path = candidate
                    break

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found.")
    return FileResponse(path)

app.include_router(tracking_router)


@app.get("/health")
def health_check():
    data_dirs = {"data/users": DATA_ROOT.exists()}
    return {
        "status": "ok" if all(data_dirs.values()) else "degraded",
        "jobs": runtime.job_manager.summary(),
        "model_cache": model_cache.status(),
        "data_dirs": data_dirs,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    job_summary = runtime.job_manager.summary()
    cache_status = model_cache.status()
    lines = [
        "# HELP mot_reid_jobs_total Jobs tracked by status.",
        "# TYPE mot_reid_jobs_total gauge",
    ]
    for status_name in ("queued", "running", "cancel_requested", "canceled", "failed", "completed"):
        lines.append(f'mot_reid_jobs_total{{status="{status_name}"}} {int(job_summary.get(status_name, 0))}')
    lines.extend(
        [
            "# HELP mot_reid_model_cache_entries Cached model objects.",
            "# TYPE mot_reid_model_cache_entries gauge",
            f'mot_reid_model_cache_entries{{type="detector"}} {int(cache_status.get("detectors", 0))}',
            f'mot_reid_model_cache_entries{{type="reid"}} {int(cache_status.get("reid_encoders", 0))}',
            f'mot_reid_model_cache_entries{{type="clip"}} {int(cache_status.get("clip_models", 0))}',
        ]
    )
    return "\n".join(lines) + "\n"
