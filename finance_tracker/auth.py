from __future__ import annotations

import hmac

import streamlit as st

import bcrypt

from finance_tracker.config import get_config


def check_password() -> bool:
    """
    Password gate.

    - Prefer bcrypt verification via `password_hash` from secrets/env.
    - Fallback to plaintext verification via `password` from secrets/env (dev only).
    """
    cfg = get_config()

    if not cfg.password_hash and not cfg.password:
        st.error("Auth not configured. Set `password_hash` (recommended) or `password` in Streamlit secrets.")
        return False

    def password_entered() -> None:
        raw = st.session_state.get("password") or ""

        if cfg.password_hash:
            try:
                st.session_state["password_correct"] = bcrypt.checkpw(
                    raw.encode("utf-8"), cfg.password_hash.encode("utf-8")
                )
            except Exception:
                st.session_state["password_correct"] = False
        else:
            # Dev/local fallback: constant-time comparison.
            st.session_state["password_correct"] = hmac.compare_digest(raw, str(cfg.password))

        if st.session_state["password_correct"]:
            # Remove password after verification to reduce accidental leakage.
            st.session_state.pop("password", None)

    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        return False

    if not st.session_state["password_correct"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False

    return True

