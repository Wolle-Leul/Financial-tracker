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

