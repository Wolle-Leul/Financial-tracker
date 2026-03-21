"""
Polish bank card-statement layout: two ISO dates, PŁATNOŚĆ KARTĄ, value/balance, merchant, Karta.

Uses value date (second date) as txn_datetime. Card payment amounts are stored as negative outflows.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Union

from finance_tracker.parsers.generic_pdf_parser import (
    ParseWarning,
    _extract_text_from_pdf,
    _parse_amount,
    _parse_date,
)

# Posting date, value date, card payment label, transaction value, balance, description ending with Karta.
# Amounts use Polish formatting (comma decimal); whitespace is normalized before matching.
_POLISH_CARD_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})\s+PŁATNOŚĆ\s+KARTĄ\s+"
    r"(\d[\d.,]*)\s*-\s*(\d[\d.,]*)\s+(.*?)\s+Karta\b",
    re.IGNORECASE | re.UNICODE,
)


def _normalize_pdf_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_polish_card_pdf_transactions(
    pdf_bytes: bytes,
    max_transactions: int = 500,
    return_errors: bool = False,
) -> Union[List[Dict[str, object]], Tuple[List[Dict[str, object]], List[ParseWarning]]]:
    """
    Parse Polish card-payment lines from full PDF text. Returns the same dict shape as
    ``parse_pdf_transactions`` from generic_pdf_parser.
    """
    text = _extract_text_from_pdf(pdf_bytes)
    normalized = _normalize_pdf_text(text)
    matches = _POLISH_CARD_PATTERN.findall(normalized)

    results: List[Dict[str, object]] = []
    warnings: List[ParseWarning] = []

    for groups in matches:
        post_raw, value_raw, value_amt_raw, _balance_raw, desc = groups

        # Value date = booking date for the transaction.
        dt = _parse_date(value_raw)
        if dt is None:
            warnings.append(ParseWarning(reason="Unparseable value date", raw_line=value_raw))
            continue

        amt = _parse_amount(value_amt_raw)
        if amt is None:
            warnings.append(ParseWarning(reason="Unparseable transaction amount", raw_line=value_amt_raw))
            continue

        # Card payments are outflows; balance column is informational only.
        if amt > 0:
            amt = -amt
        elif amt == 0:
            warnings.append(ParseWarning(reason="Amount is zero; skipping", raw_line=value_amt_raw))
            continue

        desc_clean = desc.strip()
        results.append(
            {
                "txn_datetime": dt,
                "amount": amt,
                "merchant": None,
                "description": f"{desc_clean} (post. {post_raw})",
            }
        )

        if len(results) >= max_transactions:
            break

    if return_errors:
        if not results and not warnings:
            warnings.append(
                ParseWarning(
                    reason="No Polish card-payment lines matched (PŁATNOŚĆ KARTĄ … Karta).",
                    raw_line=normalized[:400],
                )
            )
        return results, warnings

    return results
