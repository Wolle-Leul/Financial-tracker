from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple, Union

_DATE_PATTERNS = [
    # dd.mm.yyyy or dd/mm/yyyy
    re.compile(r"(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"),
    # yyyy-mm-dd
    re.compile(r"(?P<date>\d{4}[./-]\d{1,2}[./-]\d{1,2})"),
]


def _parse_date(raw: str) -> Optional[datetime]:
    raw = raw.strip()
    # Normalize separators.
    candidate = raw.replace("/", ".").replace("-", ".")
    parts = candidate.split(".")
    if len(parts) != 3:
        return None

    a, b, c = parts
    try:
        if len(a) == 4:
            # yyyy.mm.dd
            year = int(a)
            month = int(b)
            day = int(c)
            return datetime(year, month, day)

        # dd.mm.yyyy or dd.mm.yy
        day = int(a)
        month = int(b)
        year = int(c) if len(c) == 4 else int("20" + c)
        return datetime(year, month, day)
    except Exception:
        return None
    return None


def _parse_amount(raw: str) -> Optional[Decimal]:
    """
    Parse common finance formats:
    - -1,234.56
    - -1.234,56
    - 1234.56
    - 1234,56
    """
    s = raw.strip()
    if not s:
        return None

    # Remove currency symbols and spaces.
    s = re.sub(r"[^0-9,.\-]", "", s)
    if not s or s in {".", ","}:
        return None

    # Determine decimal separator based on last occurrence.
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")
    decimal_sep = None
    if last_comma > last_dot:
        decimal_sep = ","
    elif last_dot > last_comma:
        decimal_sep = "."
    else:
        # No clear separator, parse as integer-like.
        decimal_sep = None

    try:
        if decimal_sep == ",":
            # thousands sep are dots
            s = s.replace(".", "")
            s = s.replace(",", ".")
        elif decimal_sep == ".":
            s = s  # thousands sep are commas
            s = s.replace(",", "")
        else:
            s = s.replace(",", "")

        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


_LINE_REGEXES = [
    # Date Amount Description
    re.compile(
        r"^(?P<date>\d{4}[./-]\d{1,2}[./-]\d{1,2})\s+(?P<amount>-?\d[\d\s.,]*)\s+(?P<desc>.+)"
    ),
    re.compile(
        r"^(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s+(?P<amount>-?\d[\d\s.,]*)\s+(?P<desc>.+)"
    ),
]


@dataclass(frozen=True)
class ParseWarning:
    reason: str
    raw_line: str


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import PyPDF2  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyPDF2 is required for PDF text extraction. Install it (see requirements.txt)."
        ) from e

    reader = PyPDF2.PdfReader(pdf_bytes)
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def parse_pdf_transactions(
    pdf_bytes: bytes,
    max_transactions: int = 500,
    return_errors: bool = False,
) -> Union[List[Dict[str, object]], Tuple[List[Dict[str, object]], List[ParseWarning]]]:
    """
    Extract -> detect -> normalize -> validate transaction rows (heuristics).

    When `return_errors=True`, returns `(transactions, warnings)` where warnings
    are emitted for parse/normalization failures and unmatched patterns.
    """
    text = _extract_text_from_pdf(pdf_bytes)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    results: List[Dict[str, object]] = []
    warnings: List[ParseWarning] = []

    for line in lines:
        matched_any = False
        for pat in _LINE_REGEXES:
            m = pat.search(line)
            if not m:
                continue

            matched_any = True

            raw_date = m.group("date")
            raw_amount = m.group("amount")
            desc = m.groupdict().get("desc") or ""

            # Normalize
            dt = _parse_date(raw_date)
            amt = _parse_amount(raw_amount)

            # Validate
            if dt is None:
                warnings.append(ParseWarning(reason="Unparseable date", raw_line=line))
                break
            if amt is None:
                warnings.append(ParseWarning(reason="Unparseable amount", raw_line=line))
                break
            if amt == 0:
                warnings.append(ParseWarning(reason="Amount is zero; skipping", raw_line=line))
                break

            results.append(
                {
                    "txn_datetime": dt,
                    "amount": amt,
                    "merchant": None,
                    "description": desc.strip(),
                }
            )

            if len(results) >= max_transactions:
                if return_errors:
                    return results, warnings
                return results

            break

        if not matched_any:
            continue

    if return_errors:
        if not results:
            warnings.append(
                ParseWarning(
                    reason="No transactions matched known patterns. Try adding a bank-specific template.",
                    raw_line=text[:300].replace("\n", " "),
                )
            )
        return results, warnings

    return results

