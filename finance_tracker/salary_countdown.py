"""Human-readable countdown line from configured income sources (per pay day + label)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from finance_tracker.salary import compute_salary_dates


@dataclass(frozen=True)
class SalaryCountdownInfo:
    """Nearest next pay date among configured income streams (or global default)."""

    days: int
    due_date: datetime
    stream_label: str

    @property
    def label_text(self) -> str:
        return f"{max(0, self.days)} days until {self.stream_label}"


def compute_salary_countdown(
    income_rows: List[Tuple[str, Optional[int]]],
    *,
    reference_date: datetime,
    holidays: List[str],
    default_pay_day: int,
) -> SalaryCountdownInfo:
    """
    Pick the *nearest* next pay date among income sources.

    Each row is ``(label, salary_day_of_month)`` where ``salary_day_of_month`` is
    None → use ``default_pay_day`` (usually from global salary_rules).

    If ``income_rows`` is empty, fall back to a single schedule using ``default_pay_day``.
    """
    dom_default = max(1, min(31, int(default_pay_day)))

    if not income_rows:
        _, due = compute_salary_dates(reference_date, holidays, salary_day_of_month=dom_default)
        days = (due - reference_date).days
        return SalaryCountdownInfo(days=max(0, days), due_date=due, stream_label="next pay")

    best_due: datetime | None = None
    best_days = 0
    best_label = "Income"

    for label, dom_opt in income_rows:
        dom = int(dom_opt) if dom_opt is not None else dom_default
        dom = max(1, min(31, dom))
        _, due = compute_salary_dates(reference_date, holidays, salary_day_of_month=dom)
        if best_due is None or due < best_due:
            best_due = due
            best_days = (due - reference_date).days
            best_label = (label or "Income").strip() or "Income"

    assert best_due is not None
    return SalaryCountdownInfo(days=max(0, best_days), due_date=best_due, stream_label=best_label)


def compute_salary_countdown_label(
    income_rows: List[Tuple[str, Optional[int]]],
    *,
    reference_date: datetime,
    holidays: List[str],
    default_pay_day: int,
) -> str:
    return compute_salary_countdown(
        income_rows,
        reference_date=reference_date,
        holidays=holidays,
        default_pay_day=default_pay_day,
    ).label_text
