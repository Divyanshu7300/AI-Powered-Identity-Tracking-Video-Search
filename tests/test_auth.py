from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.auth import LoginRequest, SignupRequest, _decode, login, signup


def test_login_token_contains_private_session_namespace(monkeypatch) -> None:
    monkeypatch.setenv("MOT_REID_AUTH_SECRET", "test-secret-with-enough-entropy")
    monkeypatch.setenv("MOT_REID_AUTH_USERNAME", "operator")
    monkeypatch.setenv("MOT_REID_AUTH_PASSWORD", "correct-password")

    result = login(LoginRequest(username="operator", password="correct-password"), "auth-test")
    identity = _decode(result["access_token"])

    assert identity["sub"] == "operator"
    assert identity["sid"]
    assert identity["sid"] != "auth-test"


def test_login_rejects_invalid_credentials(monkeypatch) -> None:
    monkeypatch.setenv("MOT_REID_AUTH_SECRET", "test-secret-with-enough-entropy")
    monkeypatch.setenv("MOT_REID_AUTH_USERNAME", "operator")
    monkeypatch.setenv("MOT_REID_AUTH_PASSWORD", "correct-password")

    with pytest.raises(HTTPException) as error:
        login(LoginRequest(username="operator", password="wrong-password"), "auth-invalid-test")

    assert error.value.status_code == 401


def test_signup_creates_an_account_that_can_log_in(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MOT_REID_AUTH_SECRET", "test-secret-with-enough-entropy")
    monkeypatch.setenv("MOT_REID_AUTH_DB", str(tmp_path / "users.sqlite3"))
    monkeypatch.setenv("MOT_REID_ALLOW_SIGNUP", "true")

    created = signup(SignupRequest(username="new.operator", password="a-secure-password", password_confirmation="a-secure-password"))
    logged_in = login(LoginRequest(username="new.operator", password="a-secure-password"), "signup-login-test")

    assert _decode(created["access_token"])["sub"] == "new.operator"
    assert _decode(logged_in["access_token"])["sub"] == "new.operator"
