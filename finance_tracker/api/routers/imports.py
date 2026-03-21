from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.models import Category, SubCategory, Transaction
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.dashboard import (
    ImportPdfResponse,
    MapTransactionRequest,
    SubcategoryOption,
    UncategorizedItem,
    UncategorizedResponse,
)
from finance_tracker.services.import_service import import_statement_pdf
from sqlalchemy import select

from finance_tracker.db.session import session_scope

router = APIRouter(prefix="/api", tags=["imports"])


@router.post("/imports/pdf", response_model=ImportPdfResponse)
async def post_import_pdf(
    request: Request,
    file: UploadFile = File(...),
) -> ImportPdfResponse:
    require_session_auth(request)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        result = import_statement_pdf(data, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ImportPdfResponse(**result)


@router.get("/imports/{import_id}/uncategorized", response_model=UncategorizedResponse)
def get_uncategorized(
    import_id: int,
    request: Request,
) -> UncategorizedResponse:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    with session_scope() as session:
        uncategorized_stmt = (
            select(Transaction.id, Transaction.description)
            .where(Transaction.user_id == user_id)
            .where(Transaction.import_id == import_id)
            .where(Transaction.subcategory_id.is_(None))
            .limit(50)
        )
        uncategorized = session.execute(uncategorized_stmt).all()

        subcats_stmt = (
            select(SubCategory.id, SubCategory.category_id, SubCategory.name, Category.name)
            .join(Category, SubCategory.category_id == Category.id)
            .where(Category.user_id == user_id)
        )
        subcats = session.execute(subcats_stmt).all()

    transactions = [
        UncategorizedItem(id=int(r[0]), label=f"{int(r[0])}: {(r[1] or '')[:45]}")
        for r in uncategorized
    ]
    subcategories = [
        SubcategoryOption(id=int(r[0]), label=f"{r[3]} - {r[2]}")
        for r in subcats
    ]
    return UncategorizedResponse(transactions=transactions, subcategories=subcategories)


@router.patch("/transactions/{transaction_id}")
def patch_transaction(
    transaction_id: int,
    body: MapTransactionRequest,
    request: Request,
) -> dict[str, bool]:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    subcat_id = int(body.subcategory_id)

    with session_scope() as session:
        subcat_row = session.execute(
            select(SubCategory.category_id, SubCategory.id)
            .join(Category, SubCategory.category_id == Category.id)
            .where(Category.user_id == user_id)
            .where(SubCategory.id == subcat_id)
        ).first()
        if not subcat_row:
            raise HTTPException(status_code=400, detail="Invalid subcategory for user")

        category_id = int(subcat_row[0])

        txn = session.execute(
            select(Transaction).where(Transaction.user_id == user_id, Transaction.id == transaction_id)
        ).scalar_one_or_none()
        if txn is None:
            raise HTTPException(status_code=404, detail="Transaction not found")

        txn.category_id = category_id
        txn.subcategory_id = subcat_id

    return {"ok": True}
