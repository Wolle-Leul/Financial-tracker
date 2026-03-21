from __future__ import annotations

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.requests import Request

from finance_tracker.config import get_session_secret_value

_MAX_AGE_SEC = 60 * 60 * 24 * 7  # 7 days


def get_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if len(auth) > 7 and auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return None


def create_auth_token() -> str:
    ser = URLSafeTimedSerializer(get_session_secret_value(), salt="ft-api-auth")
    return ser.dumps({"sub": "auth"})


def verify_auth_token(token: str) -> bool:
    if not token:
        return False
    ser = URLSafeTimedSerializer(get_session_secret_value(), salt="ft-api-auth")
    try:
        data = ser.loads(token, max_age=_MAX_AGE_SEC)
        return isinstance(data, dict) and data.get("sub") == "auth"
    except (BadSignature, SignatureExpired):
        return False
