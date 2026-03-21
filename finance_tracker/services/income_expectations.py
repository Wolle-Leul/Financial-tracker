from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select

from finance_tracker.db.models import IncomeSource
from finance_tracker.db.session import session_scope
from finance_tracker.tax.pl_gross_net import estimate_net_from_gross


def sum_expected_net_monthly(user_id: int) -> float:
    """
    Sum configured expected net income from all income sources for the user.
    """
    with session_scope() as session:
        rows = (
            session.execute(
                select(IncomeSource).where(IncomeSource.user_id == user_id).order_by(IncomeSource.sort_order)
            )
            .scalars()
            .all()
        )

    total = 0.0
    for r in rows:
        if r.use_net_only:
            if r.net_amount is not None:
                total += float(r.net_amount)
            continue
        if r.gross_amount is not None:
            est = estimate_net_from_gross(float(r.gross_amount), r.contract_type)
            total += est.net
        elif r.net_amount is not None:
            total += float(r.net_amount)
    return total
