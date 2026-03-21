from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from finance_tracker.db.models import Category, Import as ImportModel
from finance_tracker.db.models import SubCategory, Transaction
from finance_tracker.db.session import init_db_for_dev, session_scope
from finance_tracker.parsers.generic_pdf_parser import parse_pdf_transactions
from finance_tracker.parsers.polish_card_statement import parse_polish_card_pdf_transactions
from finance_tracker.db.seed import ensure_demo_categories_seeded
from finance_tracker.db.user import get_or_create_default_user_id


def _map_description_to_subcategory(
    haystack: str,
    subcategories: list[tuple[int, int, str | None]],
) -> tuple[int | None, int | None]:
    """
    Return `(category_id, subcategory_id)` for the first keyword match.
    """
    haystack = haystack.lower()
    for category_id, subcategory_id, keywords in subcategories:
        if not keywords:
            continue
        for kw in keywords.split(","):
            kw = kw.strip().lower()
            if not kw:
                continue
            if kw in haystack:
                return category_id, subcategory_id
    return None, None


def import_statement_pdf(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Parse a bank statement PDF and persist:
    - one `imports` row
    - N `transactions` rows

    Notes:
    - Category/subcategory mapping is intentionally left null for now.
      It will be improved in later to-dos (category-mapping).
    """
    # Ensure local dev has tables. In production you'd typically run Alembic.
    try:
        init_db_for_dev()
    except Exception:
        # If migrations are required and create_all fails, we'll surface later during insert.
        pass

    user_id = get_or_create_default_user_id()

    # Ensure category/subcategory records exist (and have keywords for mapping).
    ensure_demo_categories_seeded(user_id)

    generic_txns, generic_warnings = parse_pdf_transactions(pdf_bytes, return_errors=True)
    polish_warnings: List[Any] = []
    if generic_txns:
        transactions_parsed, parse_warnings = generic_txns, generic_warnings
    else:
        polish_txns, polish_warnings = parse_polish_card_pdf_transactions(
            pdf_bytes, return_errors=True
        )
        if polish_txns:
            transactions_parsed, parse_warnings = polish_txns, polish_warnings
        else:
            transactions_parsed, parse_warnings = [], generic_warnings

    if not transactions_parsed:
        hint_parts: List[str] = []
        if generic_warnings:
            hint_parts.append(f"generic: {generic_warnings[0].reason}")
        if polish_warnings:
            hint_parts.append(f"polish: {polish_warnings[0].reason}")
        hint = (
            "; ".join(hint_parts)
            if hint_parts
            else "Unknown parsing failure (generic and Polish card template found nothing)"
        )
        raise ValueError(f"No transactions parsed from PDF. Parser hint: {hint}")

    with session_scope() as session:
        # Load subcategory keyword rules for mapping.
        subcats_stmt = (
            select(SubCategory.category_id, SubCategory.id, SubCategory.match_keywords)
            .join(Category, SubCategory.category_id == Category.id)
            .where(Category.user_id == user_id)
        )
        subcats = [(int(r[0]), int(r[1]), r[2]) for r in session.execute(subcats_stmt).all()]

        imp = ImportModel(
            user_id=user_id,
            source_filename=filename,
            imported_at=datetime.utcnow(),
            rows_parsed=len(transactions_parsed),
            parse_warnings_count=len(parse_warnings),
            categorized_txn_count=0,
            uncategorized_txn_count=0,
        )
        session.add(imp)
        session.flush()

        rows_added = 0
        categorized_count = 0
        uncategorized_count = 0
        for t in transactions_parsed:
            desc = str(t.get("description") or "")
            merch = str(t.get("merchant") or "")
            category_id, subcategory_id = _map_description_to_subcategory(
                haystack=f"{merch} {desc}",
                subcategories=subcats,
            )
            txn = Transaction(
                user_id=user_id,
                import_id=imp.id,
                txn_datetime=t["txn_datetime"],
                merchant=t.get("merchant"),
                description=t.get("description"),
                amount=Decimal(t["amount"]),
                currency="PLN",
                category_id=category_id,
                subcategory_id=subcategory_id,
                raw_text=t.get("description"),
            )
            session.add(txn)
            rows_added += 1
            if subcategory_id is None:
                uncategorized_count += 1
            else:
                categorized_count += 1

        imp.categorized_txn_count = categorized_count
        imp.uncategorized_txn_count = uncategorized_count

        # session_scope commits on success
        return {
            "import_id": int(imp.id),
            "rows_added": rows_added,
            "parsed_total": len(transactions_parsed),
            "parse_warnings_count": len(parse_warnings),
            "categorized_txn_count": categorized_count,
            "uncategorized_txn_count": uncategorized_count,
        }

