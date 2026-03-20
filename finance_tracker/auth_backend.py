from __future__ import annotations

import hmac

import bcrypt

from finance_tracker.config import get_config


def verify_password(raw: str) -> bool:
    """
    Verify the dashboard password (bcrypt hash or dev plaintext).
    Used by Streamlit and the FastAPI auth routes.
    """
    cfg = get_config()

    if not cfg.password_hash and not cfg.password:
        return False

    if cfg.password_hash:
        try:
            return bcrypt.checkpw(raw.encode("utf-8"), cfg.password_hash.encode("utf-8"))
        except Exception:
            return False

    return hmac.compare_digest(raw, str(cfg.password))
