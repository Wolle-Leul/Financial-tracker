from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

from finance_tracker.auth_backend import verify_password
from finance_tracker.config import get_config
from finance_tracker.schemas.dashboard import LoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, bool]:
    cfg = get_config()
    if not cfg.password_hash and not cfg.password:
        raise HTTPException(
            status_code=503,
            detail="Auth not configured. Set PASSWORD_HASH or PASSWORD in the environment.",
        )
    if not verify_password(body.password):
        raise HTTPException(status_code=401, detail="Incorrect password")
    request.session["authenticated"] = True
    return {"ok": True}


@router.post("/logout")
def logout(request: Request) -> dict[str, bool]:
    request.session.clear()
    return {"ok": True}


@router.get("/me")
def me(request: Request) -> dict[str, bool]:
    return {"authenticated": bool(request.session.get("authenticated"))}
