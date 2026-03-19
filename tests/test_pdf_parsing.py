from __future__ import annotations

from datetime import datetime

from finance_tracker.parsers import generic_pdf_parser as gp


def test_parse_pdf_transactions_with_mock_extraction(monkeypatch) -> None:
    sample_text = "\n".join(
        [
            "03.04.2026 10,50 Groceries supermarket",
            "2026-04-04 -120.00 Rent landlord",
        ]
    )

    def fake_extract_text(_pdf_bytes: bytes) -> str:
        return sample_text

    monkeypatch.setattr(gp, "_extract_text_from_pdf", fake_extract_text)

    txns, warnings = gp.parse_pdf_transactions(b"dummy", return_errors=True)
    assert warnings == []
    assert len(txns) == 2

    assert txns[0]["txn_datetime"] == datetime(2026, 4, 3)
    assert float(txns[0]["amount"]) == 10.50
    assert "Groceries" in txns[0]["description"]

    assert txns[1]["txn_datetime"] == datetime(2026, 4, 4)
    assert float(txns[1]["amount"]) == -120.00

