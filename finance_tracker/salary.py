from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple


def generate_salaryday(
    year: int,
    month: int,
    holidays: List[str],
    salary_date: int = 10,
) -> datetime:
    """
    Compute salary "due" date adjusted for weekends/holidays.

    Mirrors the original app logic:
    - Default salary day is the `salary_date` day-of-month
    - If it falls on weekend/weekday holiday, shift backwards by 1-3 days
    """
    salary_date_dt = datetime(year, month, salary_date)

    if salary_date_dt.weekday() < 5:  # Mon-Fri
        if salary_date_dt.strftime("%Y-%m-%d") in holidays:
            # If holiday: Monday shifts 3 days, other weekdays shift 1 day.
            if salary_date_dt.weekday() == 0:
                salary_date_dt -= timedelta(days=3)
            else:
                salary_date_dt -= timedelta(days=1)
    else:  # weekend
        if salary_date_dt.weekday() == 5:  # Saturday
            salary_date_dt -= timedelta(days=1)
        else:  # Sunday
            salary_date_dt -= timedelta(days=2)

    return salary_date_dt


def compute_salary_dates(today: datetime, holidays: List[str]) -> Tuple[datetime, datetime]:
    """
    Return `(salary_prev_month, due_salary_date)` relative to `today`.

    This is a direct extraction of the logic from `app.py` to make it testable
    and reusable in metrics/calendar rendering.
    """
    today_year = today.year
    today_month = today.month
    today_date = today.day

    salary_this_month = generate_salaryday(today_year, today_month, holidays)

    # Previous salary date
    if salary_this_month.month == today_month:
        if salary_this_month.day <= today_date:
            salary_prev_month = salary_this_month
        elif today_month == 1:
            salary_prev_month = generate_salaryday(today_year - 1, 12, holidays)
        else:
            salary_prev_month = generate_salaryday(today_year, today_month - 1, holidays)
    else:
        salary_prev_month = generate_salaryday(today_year, today_month - 1, holidays)

    # Due salary date
    if salary_this_month.month == today_month:
        if salary_this_month.day <= today_date:
            if today_month == 12:
                due_salary_date = generate_salaryday(today_year + 1, 1, holidays)
            else:
                due_salary_date = generate_salaryday(today_year, today_month + 1, holidays)
        else:
            due_salary_date = salary_this_month
    else:
        due_salary_date = generate_salaryday(today_year, today_month + 1, holidays)

    return salary_prev_month, due_salary_date

