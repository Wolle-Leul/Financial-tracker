from __future__ import annotations

from decimal import Decimal
from datetime import datetime

from finance_tracker.parsers import generic_pdf_parser as gp


def test_parse_date_dd_mm_yyyy() -> None:
    dt = gp._parse_date("03.04.2026")
    assert dt == datetime(2026, 4, 3)


def test_parse_date_iso() -> None:
    dt = gp._parse_date("2026-04-03")
    assert dt == datetime(2026, 4, 3)


def test_parse_amount_comma_decimal() -> None:
    amt = gp._parse_amount("1.234,56")
    assert amt == Decimal("1234.56")


def test_parse_amount_dot_decimal() -> None:
    amt = gp._parse_amount("1,234.56")
    assert amt == Decimal("1234.56")


def test_parse_amount_negative() -> None:
    amt = gp._parse_amount("-10,00")
    assert amt == Decimal("-10.00")


def test_parse_amount_invalid_returns_none() -> None:
    amt = gp._parse_amount("not-a-number")
    assert amt is None

