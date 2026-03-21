from __future__ import annotations

import hmac

import bcrypt

from finance_tracker.config import get_config


def verify_password(password: str) -> bool:
    """API login: bcrypt hash or plaintext (dev) from config."""
    cfg = get_config()
    if not cfg.password_hash and not cfg.password:
        return False
    if cfg.password_hash:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), cfg.password_hash.encode("utf-8"))
        except Exception:
            return False
    return hmac.compare_digest(password, str(cfg.password))
