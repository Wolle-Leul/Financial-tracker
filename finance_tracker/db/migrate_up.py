"""
Run Alembic migrations to head (used on API startup for hosted deploys).

Set SKIP_DB_MIGRATIONS=1 to disable (e.g. local SQLite experiments).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config

from finance_tracker.config import get_config

_log = logging.getLogger("uvicorn.error")


def migrate_upgrade() -> None:
    if os.getenv("SKIP_DB_MIGRATIONS", "").lower() in ("1", "true", "yes"):
        _log.info("SKIP_DB_MIGRATIONS set — skipping Alembic upgrade")
        return

    # finance_tracker/db/migrate_up.py -> repo root is parents[2]
    root = Path(__file__).resolve().parents[2]
    ini = root / "alembic.ini"
    if not ini.is_file():
        _log.warning("alembic.ini not found at %s — skipping migrations", ini)
        return

    cfg = Config(str(ini))
    # ConfigParser treats % as interpolation; passwords may contain %2F etc.
    raw_url = get_config().db_url or ""
    cfg.set_main_option("sqlalchemy.url", raw_url.replace("%", "%%"))
    _log.info("Running Alembic upgrade head…")
    command.upgrade(cfg, "head")
    _log.info("Alembic upgrade head completed")
