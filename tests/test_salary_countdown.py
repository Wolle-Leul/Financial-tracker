from datetime import datetime

from finance_tracker.salary_countdown import compute_salary_countdown


def test_countdown_empty_rows_uses_default_dom():
    ref = datetime(2025, 3, 10, 12, 0, 0)
    info = compute_salary_countdown(
        [],
        reference_date=ref,
        holidays=[],
        default_pay_day=25,
    )
    assert info.stream_label == "next pay"
    assert info.days >= 0
    assert "days until next pay" in info.label_text


def test_countdown_picks_earliest_due_among_streams():
    ref = datetime(2025, 3, 10, 12, 0, 0)
    info = compute_salary_countdown(
        [("Soon", 15), ("Later", 28)],
        reference_date=ref,
        holidays=[],
        default_pay_day=25,
    )
    assert info.stream_label == "Soon"
    assert "Soon" in info.label_text
