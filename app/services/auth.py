from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from threading import Lock

from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=256)


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=4, max_length=256)
    password_confirmation: str = Field(..., min_length=1, max_length=256)


_login_attempts: dict[str, list[float]] = {}
_login_lock = Lock()
LOGIN_ATTEMPTS_PER_MINUTE = int(os.getenv("MOT_REID_LOGIN_ATTEMPTS_PER_MINUTE", "5"))
PASSWORD_HASH_ITERATIONS = 600_000


def _user_database() -> sqlite3.Connection:
    database_path = os.getenv("MOT_REID_AUTH_DB", "data/auth/users.sqlite3")
    path = os.path.abspath(database_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            email TEXT
        )"""
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(users)")}
    if "email" not in columns:
        connection.execute("ALTER TABLE users ADD COLUMN email TEXT")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_email_unique ON users(email) WHERE email IS NOT NULL")
    return connection


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_HASH_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def _password_matches(password: str, stored_value: str) -> bool:
    try:
        algorithm, iterations, encoded_salt, encoded_digest = stored_value.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(encoded_digest.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False


def _record_auth_attempt(client_key: str) -> None:
    now = time.time()
    with _login_lock:
        attempts = [stamp for stamp in _login_attempts.get(client_key, []) if now - stamp < 60]
        if len(attempts) >= LOGIN_ATTEMPTS_PER_MINUTE:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many authentication attempts. Try again later.")
        attempts.append(now)
        _login_attempts[client_key] = attempts


def _session_result(username: str) -> dict:
    token = _encode({"sub": username, "sid": secrets.token_urlsafe(24), "exp": int(time.time()) + 8 * 3600})
    return {"access_token": token, "token_type": "bearer", "expires_in": 8 * 3600, "username": username}


def _secret() -> bytes:
    value = os.getenv("MOT_REID_AUTH_SECRET", "")
    if not value:
        raise RuntimeError("MOT_REID_AUTH_SECRET must be configured before enabling authentication.")
    return value.encode("utf-8")


def _encode(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    signature = hmac.new(_secret(), body, hashlib.sha256).digest()
    return f"{body.decode()}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def _decode(token: str) -> dict:
    try:
        body, encoded_signature = token.split(".", 1)
        expected = hmac.new(_secret(), body.encode(), hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode(encoded_signature + "=" * (-len(encoded_signature) % 4))
        if not hmac.compare_digest(expected, supplied):
            raise ValueError
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if payload.get("exp", 0) < int(time.time()) or not payload.get("sid"):
            raise ValueError
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired login token.")


def login(payload: LoginRequest, client_key: str = "unknown") -> dict:
    _record_auth_attempt(client_key)
    with _user_database() as database:
        row = database.execute("SELECT password_hash FROM users WHERE username = ?", (payload.username,)).fetchone()
    valid_registered_user = row is not None and _password_matches(payload.password, row[0])
    username = os.getenv("MOT_REID_AUTH_USERNAME", "admin")
    password = os.getenv("MOT_REID_AUTH_PASSWORD", "")
    valid_environment_user = bool(password) and hmac.compare_digest(payload.username, username) and hmac.compare_digest(payload.password, password)
    if not valid_registered_user and not valid_environment_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")
    return _session_result(payload.username)


def signup(payload: SignupRequest, client_key: str = "unknown") -> dict:
    _record_auth_attempt(client_key)
    if os.getenv("MOT_REID_ALLOW_SIGNUP", "false").lower() != "true":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="New account registration is disabled.")
    if not hmac.compare_digest(payload.password, payload.password_confirmation):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Passwords do not match.")
    with _user_database() as database:
        try:
            database.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (payload.username, _hash_password(payload.password), int(time.time())),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="That username is already in use.")
    return _session_result(payload.username)


def current_identity(request: Request) -> dict:
    proxy_secret = os.getenv("MOT_REID_INTERNAL_PROXY_SECRET", "") or os.getenv("BACKEND_INTERNAL_PROXY_SECRET", "")
    proxy_user_id = request.headers.get("x-mot-reid-user-id", "")
    if proxy_secret and proxy_user_id and hmac.compare_digest(request.headers.get("x-mot-reid-proxy-secret", ""), proxy_secret):
        identity = {"sub": proxy_user_id, "sid": proxy_user_id}
        request.state.identity = identity
        return identity
    authorization = request.headers.get("authorization", "")
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else request.cookies.get("mot_reid_access_token", "")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required.", headers={"WWW-Authenticate": "Bearer"})
    identity = _decode(token)
    request.state.identity = identity
    return identity


def require_auth(identity: dict = Depends(current_identity)) -> dict:
    return identity
