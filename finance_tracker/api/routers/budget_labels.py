"""Create/delete subcategories (budget lines) for mapping and recurring bills."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.models import Category, SubCategory
from finance_tracker.db.session import session_scope
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.settings import (
    BudgetCategoryOut,
    BudgetLabelsResponse,
    SubcategoryCreateBody,
    SubcategoryCreatedOut,
)

router = APIRouter(prefix="/api", tags=["budget-labels"])


@router.get("/budget-labels", response_model=BudgetLabelsResponse)
def list_budget_labels(request: Request) -> BudgetLabelsResponse:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        cats = session.execute(select(Category).where(Category.user_id == user_id).order_by(Category.name)).scalars().all()
        out: list[BudgetCategoryOut] = []
        for c in cats:
            subs = session.execute(
                select(SubCategory).where(SubCategory.category_id == c.id).order_by(SubCategory.name)
            ).scalars().all()
            out.append(
                BudgetCategoryOut(
                    id=int(c.id),
                    name=str(c.name),
                    subcategories=[
                        SubcategoryCreatedOut(id=int(s.id), category_id=int(c.id), name=str(s.name)) for s in subs
                    ],
                )
            )
    return BudgetLabelsResponse(categories=out)


@router.post("/subcategories", response_model=SubcategoryCreatedOut)
def create_subcategory(request: Request, body: SubcategoryCreateBody) -> SubcategoryCreatedOut:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        cat = session.execute(
            select(Category).where(Category.id == body.category_id, Category.user_id == user_id)
        ).scalar_one_or_none()
        if cat is None:
            raise HTTPException(status_code=404, detail="Category not found")
        sc = SubCategory(
            category_id=int(cat.id),
            name=body.name.strip(),
            match_keywords=body.match_keywords.strip() if body.match_keywords else None,
        )
        session.add(sc)
        try:
            session.flush()
        except IntegrityError as e:
            raise HTTPException(status_code=409, detail="Subcategory name already exists in this category") from e
        return SubcategoryCreatedOut(id=int(sc.id), category_id=int(cat.id), name=str(sc.name))


@router.delete("/subcategories/{subcategory_id}")
def delete_subcategory(request: Request, subcategory_id: int) -> dict[str, str]:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        row = session.execute(
            select(SubCategory)
            .join(Category, SubCategory.category_id == Category.id)
            .where(SubCategory.id == subcategory_id, Category.user_id == user_id)
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Subcategory not found")
        session.delete(row)
    return {"status": "ok"}
