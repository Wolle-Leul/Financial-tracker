from __future__ import annotations

from fastapi import HTTPException, Request

from finance_tracker.api.auth_token import get_bearer_token, verify_auth_token


def require_session_auth(request: Request) -> None:
    if request.session.get("authenticated"):
        return
    token = get_bearer_token(request)
    if token and verify_auth_token(token):
        return
    raise HTTPException(status_code=401, detail="Not authenticated")
