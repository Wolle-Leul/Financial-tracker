from __future__ import annotations

from sqlalchemy import select

from finance_tracker.config import get_config
from finance_tracker.db.models import SalaryRule
from finance_tracker.db.session import session_scope


def get_or_create_salary_rule(user_id: int) -> SalaryRule:
    """
    Return persisted salary settings for the user, seeding from AppConfig when missing.
    """
    cfg = get_config()
    with session_scope() as session:
        rule = session.execute(select(SalaryRule).where(SalaryRule.user_id == user_id)).scalar_one_or_none()
        if rule is None:
            rule = SalaryRule(
                user_id=user_id,
                salary_day_of_month=cfg.salary_day_of_month,
                holiday_country=cfg.holiday_country,
                target_ratio=cfg.target_ratio,
                budget_strategy="custom_target_ratio",
            )
            session.add(rule)
            session.flush()
        return rule
