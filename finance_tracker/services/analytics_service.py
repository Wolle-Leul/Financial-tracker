from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import List

from sqlalchemy import select

from finance_tracker.db.models import Transaction
from finance_tracker.db.session import session_scope
from finance_tracker.schemas.settings import MonthlyTrendPoint, TrendsResponse


def build_monthly_trends(user_id: int, months: int = 12) -> TrendsResponse:
    """
    Aggregate transactions by calendar month (income vs expenses vs net).
    """
    months = max(1, min(36, int(months)))
    end = datetime.utcnow()
    start = end - timedelta(days=31 * months)
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"income": 0.0, "expenses": 0.0})

    with session_scope() as session:
        rows = session.execute(
            select(Transaction.txn_datetime, Transaction.amount).where(
                Transaction.user_id == user_id,
                Transaction.txn_datetime >= start,
                Transaction.txn_datetime <= end,
            )
        ).all()

    for dt, amt_dec in rows:
        if dt is None:
            continue
        amt = float(amt_dec)
        key = f"{dt.year:04d}-{dt.month:02d}"
        if amt > 0:
            buckets[key]["income"] += amt
        else:
            buckets[key]["expenses"] += abs(amt)

    # Sort descending by month key, take last `months` distinct months
    ordered = sorted(buckets.keys(), reverse=True)[:months]
    ordered = sorted(ordered)

    points: List[MonthlyTrendPoint] = []
    for key in ordered:
        inc = buckets[key]["income"]
        exp = buckets[key]["expenses"]
        points.append(
            MonthlyTrendPoint(
                month=key,
                income=round(inc, 2),
                expenses=round(exp, 2),
                net=round(inc - exp, 2),
            )
        )

    return TrendsResponse(months=points)
