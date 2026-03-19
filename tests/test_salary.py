from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from finance_tracker.salary import compute_salary_dates, generate_salaryday


def test_generate_salaryday_holiday_weekday_shift() -> None:
    # Pick a month/year where the 10th is a weekday.
    year, month = 2026, 3
    dt = datetime(year, month, 10)
    if dt.weekday() >= 5:
        pytest.skip("Test requires a weekday 10th.")

    holidays = [dt.strftime("%Y-%m-%d")]
    expected_shift_days = 3 if dt.weekday() == 0 else 1
    adjusted = generate_salaryday(year, month, holidays, salary_date=10)
    assert adjusted == dt - timedelta(days=expected_shift_days)


def test_generate_salaryday_holiday_weekend_shift() -> None:
    # Pick a month/year where the 10th is Saturday/Sunday.
    year, month = 2026, 5
    dt = datetime(year, month, 10)
    if dt.weekday() < 5:
        pytest.skip("Test requires a weekend 10th.")

    holidays = [dt.strftime("%Y-%m-%d")]
    if dt.weekday() == 5:  # Saturday
        expected_shift_days = 1
    else:  # Sunday
        expected_shift_days = 2
    adjusted = generate_salaryday(year, month, holidays, salary_date=10)
    assert adjusted == dt - timedelta(days=expected_shift_days)


def test_compute_salary_dates_basic_invariants() -> None:
    reference_date = datetime(2026, 4, 15)
    # No holidays: should still produce prev <= reference <= due-ish.
    holidays: list[str] = []

    salary_prev_month, due_salary_date = compute_salary_dates(reference_date, holidays)
    assert salary_prev_month <= reference_date
    assert due_salary_date >= reference_date

