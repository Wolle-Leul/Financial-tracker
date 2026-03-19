from __future__ import annotations

from functools import lru_cache
from typing import List

try:
    import holidays as holiday_lib
except ModuleNotFoundError:  # pragma: no cover
    holiday_lib = None


@lru_cache(maxsize=32)
def get_holidays_poland(year: int) -> List[str]:
    """
    Return Polish holiday dates as `YYYY-MM-DD` strings.

    We keep strings because Plotly/Streamlit calendar rendering works with
    simple date lists and avoids extra datetime handling everywhere.
    """
    if holiday_lib is None:
        # If the optional `holidays` dependency isn't installed, the app should
        # still run (calendar will simply have no holiday markers).
        return []

    holiday_data = []
    for date, reason in holiday_lib.Poland(years=year).items():
        _ = reason
        holiday_data.append(date.strftime("%Y-%m-%d"))
    return sorted(holiday_data)

