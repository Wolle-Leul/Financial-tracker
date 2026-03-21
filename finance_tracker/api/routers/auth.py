from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from finance_tracker.api.auth_token import create_auth_token, get_bearer_token, verify_auth_token
from finance_tracker.auth_backend import verify_password
from finance_tracker.config import get_config
from finance_tracker.schemas.dashboard import LoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    cfg = get_config()
    if not cfg.password_hash and not cfg.password:
        raise HTTPException(
            status_code=503,
            detail="Auth not configured. Set PASSWORD_HASH or PASSWORD in the environment.",
        )
    if not verify_password(body.password):
        raise HTTPException(status_code=401, detail="Incorrect password")
    request.session["authenticated"] = True
    # Signed token for SPAs on another origin when browsers block cross-site cookies.
    return {"ok": True, "token": create_auth_token()}


@router.post("/logout")
def logout(request: Request) -> dict[str, bool]:
    request.session.clear()
    return {"ok": True}


@router.get("/me")
def me(request: Request) -> dict[str, bool]:
    if request.session.get("authenticated"):
        return {"authenticated": True}
    t = get_bearer_token(request)
    return {"authenticated": bool(t and verify_auth_token(t))}
