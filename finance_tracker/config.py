from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None


def _get_secret(key: str) -> Optional[str]:
    """
    Read configuration from Streamlit secrets (preferred) or environment variables.
    """
    # Streamlit secrets use lowercase keys by convention.
    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    # Environment variables are uppercase.
    return os.getenv(key.upper()) or os.getenv(key)


@dataclass(frozen=True)
class AppConfig:
    db_url: str

    # Authentication secrets
    password: Optional[str] = None
    password_hash: Optional[str] = None

    # Business rules
    holiday_country: str = "Poland"
    salary_day_of_month: int = 10
    target_ratio: float = 0.45


def get_config() -> AppConfig:
    db_url = _get_secret("db_url") or _get_secret("DB_URL")
    if not db_url:
        # Keep the dashboard runnable without DB until the DB feature lands.
        # Later to-dos will require this.
        db_url = "sqlite:///:memory:"

    return AppConfig(
        db_url=db_url,
        password=_get_secret("password"),
        password_hash=_get_secret("password_hash"),
        holiday_country=_get_secret("holiday_country") or "Poland",
        salary_day_of_month=int(_get_secret("salary_day_of_month") or 10),
        target_ratio=float(_get_secret("target_ratio") or 0.45),
    )


def _normalize_origin(origin: str) -> str:
    o = origin.strip()
    while o.endswith("/"):
        o = o[:-1]
    return o


def get_cors_origin_list() -> list[str]:
    """
    Browser origins allowed to call the API with credentials (cookies).
    SPA on Hostinger + API on Render must list the exact HTTPS origin(s), e.g.
    https://yoursite.hostingersite.com — trailing slashes are stripped automatically.
    """
    raw = _get_secret("cors_origins") or ""
    if not raw:
        raw = os.getenv("CORS_ORIGINS") or ""
    parts = [_normalize_origin(p) for p in str(raw).split(",") if p.strip()]
    if not parts:
        return ["http://127.0.0.1:5173", "http://localhost:5173"]
    return parts


def get_session_same_site() -> str:
    """lax | strict | none. Use none for SPA on another domain than the API (e.g. Hostinger + Render)."""
    raw = _get_secret("session_same_site") or os.getenv("SESSION_SAME_SITE") or "lax"
    s = str(raw).strip().lower()
    if s not in ("lax", "strict", "none"):
        return "lax"
    return s


def get_session_secret_value() -> str:
    secret = _get_secret("session_secret")
    if not secret:
        secret = os.getenv("SESSION_SECRET")
    if not secret:
        return "dev-session-secret-not-for-production"
    return secret

