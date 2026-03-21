from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.models import SalaryRule
from finance_tracker.db.salary_rule import get_or_create_salary_rule
from finance_tracker.db.session import session_scope
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.settings import SalaryRulePatch, SalaryRuleResponse

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/settings/salary-rule", response_model=SalaryRuleResponse)
def get_salary_rule_settings(request: Request) -> SalaryRuleResponse:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    r = get_or_create_salary_rule(user_id)
    return SalaryRuleResponse(
        salary_day_of_month=int(r.salary_day_of_month),
        holiday_country=str(r.holiday_country),
        target_ratio=float(r.target_ratio),
        budget_strategy=str(r.budget_strategy),
    )


@router.patch("/settings/salary-rule", response_model=SalaryRuleResponse)
def patch_salary_rule_settings(request: Request, body: SalaryRulePatch) -> SalaryRuleResponse:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    get_or_create_salary_rule(user_id)
    with session_scope() as session:
        r = session.execute(select(SalaryRule).where(SalaryRule.user_id == user_id)).scalar_one()
        if body.salary_day_of_month is not None:
            r.salary_day_of_month = int(body.salary_day_of_month)
        if body.holiday_country is not None:
            r.holiday_country = body.holiday_country.strip() or r.holiday_country
        if body.target_ratio is not None:
            r.target_ratio = float(body.target_ratio)
        if body.budget_strategy is not None:
            allowed = {
                "custom_target_ratio",
                "classic_50_30_20",
                "zero_based",
                "salary_window_only",
            }
            bs = body.budget_strategy.strip().lower().replace("-", "_")
            if bs not in allowed:
                raise HTTPException(status_code=400, detail=f"budget_strategy must be one of {sorted(allowed)}")
            r.budget_strategy = bs
        return SalaryRuleResponse(
            salary_day_of_month=int(r.salary_day_of_month),
            holiday_country=str(r.holiday_country),
            target_ratio=float(r.target_ratio),
            budget_strategy=str(r.budget_strategy),
        )
