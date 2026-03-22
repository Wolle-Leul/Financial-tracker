from finance_tracker.calendar_plot import build_calendar_salary_markers_for_month


def test_build_markers_includes_primary_and_streams():
    holidays: list[str] = []
    m = build_calendar_salary_markers_for_month(
        2025,
        3,
        holidays,
        default_dom=25,
        income_rows=[("Side gig", 10)],
    )
    assert isinstance(m, dict)
    assert len(m) >= 1
    for tip in m.values():
        assert "Pay —" in tip
