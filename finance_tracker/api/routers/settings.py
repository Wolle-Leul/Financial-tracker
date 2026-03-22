from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.models import Category, IncomeSource, SalaryRule, SubCategory
from finance_tracker.db.salary_rule import get_or_create_salary_rule
from finance_tracker.db.session import session_scope
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.settings import (
    SalaryRulePatch,
    SalaryRuleResponse,
    SettingsSyncRequest,
    SettingsSyncResponse,
)

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


def _allowed_strategy(bs: str) -> str:
    allowed = {
        "custom_target_ratio",
        "classic_50_30_20",
        "zero_based",
        "salary_window_only",
    }
    s = bs.strip().lower().replace("-", "_")
    if s not in allowed:
        raise HTTPException(status_code=400, detail=f"budget_strategy must be one of {sorted(allowed)}")
    return s


@router.post("/settings/sync", response_model=SettingsSyncResponse)
def sync_settings_to_database(request: Request, body: SettingsSyncRequest) -> SettingsSyncResponse:
    """
    Write pay schedule, strategy, expected income rows, and recurring bill plans to the database
    in one transaction. The dashboard reads these tables on every load (salary window, KPIs, expected net).
    """
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    get_or_create_salary_rule(user_id)

    income_updated = 0
    recurring_updated = 0

    with session_scope() as session:
        r = session.execute(select(SalaryRule).where(SalaryRule.user_id == user_id)).scalar_one()
        r.salary_day_of_month = int(body.salary_day_of_month)
        r.target_ratio = float(body.target_ratio)
        r.budget_strategy = _allowed_strategy(body.budget_strategy)

        for item in body.income_rows:
            row = session.execute(
                select(IncomeSource).where(IncomeSource.user_id == user_id, IncomeSource.id == item.id)
            ).scalar_one_or_none()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Income source id={item.id} not found")
            row.label = item.label.strip()
            row.net_amount = item.net_amount
            row.gross_amount = item.gross_amount
            income_updated += 1

        for rec in body.recurring_rows:
            sc = session.execute(
                select(SubCategory)
                .join(Category, SubCategory.category_id == Category.id)
                .where(SubCategory.id == rec.subcategory_id, Category.user_id == user_id)
            ).scalar_one_or_none()
            if sc is None:
                raise HTTPException(status_code=404, detail=f"Subcategory id={rec.subcategory_id} not found")
            if rec.planned_amount is not None:
                sc.planned_amount = Decimal(str(rec.planned_amount))
            else:
                sc.planned_amount = None
            sc.planned_deadline_day = rec.planned_deadline_day
            recurring_updated += 1

        session.flush()
        sr = session.execute(select(SalaryRule).where(SalaryRule.user_id == user_id)).scalar_one()
        salary_out = SalaryRuleResponse(
            salary_day_of_month=int(sr.salary_day_of_month),
            holiday_country=str(sr.holiday_country),
            target_ratio=float(sr.target_ratio),
            budget_strategy=str(sr.budget_strategy),
        )

    return SettingsSyncResponse(
        salary_rule=salary_out,
        income_rows_updated=income_updated,
        recurring_rows_updated=recurring_updated,
    )
