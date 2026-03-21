from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.models import Category, SubCategory
from finance_tracker.db.session import session_scope
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.settings import SubcategoryRecurringItem, SubcategoryRecurringPatch

router = APIRouter(prefix="/api", tags=["recurring"])


@router.get("/recurring-expenses", response_model=list[SubcategoryRecurringItem])
def list_recurring(request: Request) -> list[SubcategoryRecurringItem]:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        stmt = (
            select(SubCategory.id, Category.name, SubCategory.name, SubCategory.planned_amount, SubCategory.planned_deadline_day)
            .join(Category, SubCategory.category_id == Category.id)
            .where(Category.user_id == user_id)
            .order_by(Category.name, SubCategory.name)
        )
        rows = session.execute(stmt).all()
    out: list[SubcategoryRecurringItem] = []
    for rid, cname, sname, pamt, pday in rows:
        out.append(
            SubcategoryRecurringItem(
                id=int(rid),
                category_name=str(cname),
                name=str(sname),
                planned_amount=float(pamt) if pamt is not None else None,
                planned_deadline_day=int(pday) if pday is not None else None,
            )
        )
    return out


@router.patch("/recurring-expenses/{subcategory_id}", response_model=SubcategoryRecurringItem)
def patch_recurring(
    request: Request,
    subcategory_id: int,
    body: SubcategoryRecurringPatch,
) -> SubcategoryRecurringItem:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        row = session.execute(
            select(SubCategory, Category.name)
            .join(Category, SubCategory.category_id == Category.id)
            .where(Category.user_id == user_id, SubCategory.id == subcategory_id)
        ).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Subcategory not found")
        sc: SubCategory = row[0]
        cname: str = str(row[1])
        if body.planned_amount is not None:
            sc.planned_amount = Decimal(str(body.planned_amount))
        if body.planned_deadline_day is not None:
            sc.planned_deadline_day = int(body.planned_deadline_day)
        return SubcategoryRecurringItem(
            id=int(sc.id),
            category_name=cname,
            name=str(sc.name),
            planned_amount=float(sc.planned_amount) if sc.planned_amount is not None else None,
            planned_deadline_day=int(sc.planned_deadline_day) if sc.planned_deadline_day is not None else None,
        )
