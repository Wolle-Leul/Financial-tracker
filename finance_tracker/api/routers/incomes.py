from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.models import IncomeSource
from finance_tracker.db.session import session_scope
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.settings import IncomeSourceCreate, IncomeSourceItem, IncomeSourcePatch

router = APIRouter(prefix="/api", tags=["incomes"])


def _to_item(r: IncomeSource) -> IncomeSourceItem:
    return IncomeSourceItem(
        id=int(r.id),
        label=str(r.label),
        employer_name=r.employer_name,
        contract_type=str(r.contract_type),
        gross_amount=float(r.gross_amount) if r.gross_amount is not None else None,
        net_amount=float(r.net_amount) if r.net_amount is not None else None,
        use_net_only=bool(r.use_net_only),
        sort_order=int(r.sort_order),
        salary_day_of_month=int(r.salary_day_of_month) if r.salary_day_of_month is not None else None,
    )


@router.get("/income-sources", response_model=list[IncomeSourceItem])
def list_income_sources(request: Request) -> list[IncomeSourceItem]:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        rows = (
            session.execute(
                select(IncomeSource).where(IncomeSource.user_id == user_id).order_by(IncomeSource.sort_order)
            )
            .scalars()
            .all()
        )
    return [_to_item(r) for r in rows]


@router.post("/income-sources", response_model=IncomeSourceItem)
def create_income_source(request: Request, body: IncomeSourceCreate) -> IncomeSourceItem:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        row = IncomeSource(
            user_id=user_id,
            label=body.label.strip(),
            employer_name=body.employer_name.strip() if body.employer_name else None,
            contract_type=body.contract_type.strip() or "other",
            gross_amount=body.gross_amount,
            net_amount=body.net_amount,
            use_net_only=body.use_net_only,
            sort_order=int(body.sort_order),
            salary_day_of_month=body.salary_day_of_month,
        )
        session.add(row)
        session.flush()
        return _to_item(row)


@router.patch("/income-sources/{source_id}", response_model=IncomeSourceItem)
def patch_income_source(request: Request, source_id: int, body: IncomeSourcePatch) -> IncomeSourceItem:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        row = session.execute(
            select(IncomeSource).where(IncomeSource.user_id == user_id, IncomeSource.id == source_id)
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Income source not found")
        patch = body.model_dump(exclude_unset=True)
        if body.label is not None:
            row.label = body.label.strip()
        if body.employer_name is not None:
            row.employer_name = body.employer_name.strip() if body.employer_name else None
        if body.contract_type is not None:
            row.contract_type = body.contract_type.strip() or "other"
        if body.gross_amount is not None:
            row.gross_amount = body.gross_amount
        if body.net_amount is not None:
            row.net_amount = body.net_amount
        if body.use_net_only is not None:
            row.use_net_only = body.use_net_only
        if body.sort_order is not None:
            row.sort_order = int(body.sort_order)
        if "salary_day_of_month" in patch:
            row.salary_day_of_month = patch["salary_day_of_month"]
        return _to_item(row)


@router.delete("/income-sources/{source_id}")
def delete_income_source(request: Request, source_id: int) -> dict[str, bool]:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        row = session.execute(
            select(IncomeSource).where(IncomeSource.user_id == user_id, IncomeSource.id == source_id)
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Income source not found")
        session.delete(row)
    return {"ok": True}
