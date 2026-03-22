"""
Idempotent patches for production DBs that shipped code before Alembic ran.

Adds salary_rules.budget_strategy and income_sources if missing (PostgreSQL/SQLite).
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from finance_tracker.db.models import IncomeSource
from finance_tracker.db.session import get_engine

_log = logging.getLogger("uvicorn.error")


def ensure_schema_extensions() -> None:
    engine = get_engine()
    insp = inspect(engine)

    if not insp.has_table("salary_rules"):
        _log.info("ensure_schema: salary_rules missing — skipping (fresh DB uses migrations/create_all)")
        return

    cols = {c["name"] for c in insp.get_columns("salary_rules")}
    if "budget_strategy" not in cols:
        _log.warning("ensure_schema: adding salary_rules.budget_strategy (migration not applied yet)")
        dialect = engine.dialect.name
        with engine.begin() as conn:
            if dialect == "postgresql":
                conn.execute(
                    text(
                        "ALTER TABLE salary_rules ADD COLUMN IF NOT EXISTS budget_strategy "
                        "VARCHAR(64) NOT NULL DEFAULT 'custom_target_ratio'"
                    )
                )
            else:
                conn.execute(
                    text(
                        "ALTER TABLE salary_rules ADD COLUMN budget_strategy "
                        "VARCHAR(64) NOT NULL DEFAULT 'custom_target_ratio'"
                    )
                )

    if not insp.has_table("income_sources"):
        _log.warning("ensure_schema: creating income_sources table (migration not applied yet)")
        IncomeSource.__table__.create(bind=engine, checkfirst=True)
    else:
        inc_cols = {c["name"] for c in insp.get_columns("income_sources")}
        if "salary_day_of_month" not in inc_cols:
            _log.warning("ensure_schema: adding income_sources.salary_day_of_month (migration not applied yet)")
            dialect = engine.dialect.name
            with engine.begin() as conn:
                if dialect == "postgresql":
                    conn.execute(text("ALTER TABLE income_sources ADD COLUMN IF NOT EXISTS salary_day_of_month INTEGER"))
                else:
                    conn.execute(text("ALTER TABLE income_sources ADD COLUMN salary_day_of_month INTEGER"))
