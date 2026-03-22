from __future__ import annotations

import calendar
import textwrap
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .salary import generate_salaryday


def build_calendar_salary_markers_for_month(
    year: int,
    month: int,
    holidays: List[str],
    default_dom: int,
    income_rows: List[Tuple[str, Optional[int]]],
) -> Dict[int, str]:
    """
    Map calendar day-of-month → tooltip for green pay markers.

    Uses the same weekend/holiday shifting as the rest of the app. Includes the
    global default pay day plus each income stream (per-stream day or default).
    """
    day_labels: dict[int, set[str]] = {}
    seen_pairs: set[tuple[int, str]] = set()

    def push(dom: int, label: str) -> None:
        dom = max(1, min(31, int(dom)))
        dt = generate_salaryday(year, month, holidays, salary_date=dom)
        if dt.year != year or dt.month != month:
            return
        pair = (dt.day, label)
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        day_labels.setdefault(dt.day, set()).add(label)

    sd = max(1, min(31, int(default_dom)))
    push(sd, "Primary schedule")
    for lab, dom_opt in income_rows:
        dom = int(dom_opt) if dom_opt is not None else sd
        push(dom, (lab or "Income").strip() or "Income")

    out: dict[int, str] = {}
    for day in sorted(day_labels.keys()):
        labels = sorted(day_labels[day])
        if len(labels) == 1:
            out[day] = f"Pay — {labels[0]}"
        else:
            out[day] = "Pay — " + " · ".join(labels)
    return out


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def generate_calendar_html(
    year: int,
    month: int,
    holidays: List[str],
    today: datetime,
    salary_date: int = 10,
    *,
    salary_markers_by_day: Optional[Dict[int, str]] = None,
) -> str:
    """
    Premium HTML/CSS calendar grid (Mon..Sun).

    Markers:
    - Holidays: red dot + tooltip (reason if provided by caller; currently only date list is provided)
    - Salary date: green dot (one or many tooltips via ``salary_markers_by_day``)
    - Today: blue ring
    """
    first_day_of_month = datetime(year, month, 1)
    start_day = first_day_of_month.weekday()  # 0: Monday ... 6: Sunday
    num_days = calendar.monthrange(year, month)[1]
    num_cells = start_day + num_days
    rows = (num_cells + 6) // 7

    # Map holiday days for quick lookup.
    holiday_days: set[int] = set()
    for h in holidays:
        try:
            dt = datetime.strptime(h, "%Y-%m-%d")
        except Exception:
            continue
        if dt.year == year and dt.month == month:
            holiday_days.add(dt.day)

    use_marker_map = salary_markers_by_day is not None and len(salary_markers_by_day) > 0
    legacy_salary_day: Optional[int] = None
    if not use_marker_map:
        salary_dt = generate_salaryday(year, month, holidays, salary_date=salary_date)
        legacy_salary_day = salary_dt.day if salary_dt.month == month else None

    today_day: Optional[int] = today.day if today.month == month and today.year == year else None

    # Cells: index 0..rows*7-1
    cell_html: list[str] = []
    for idx in range(rows * 7):
        day_num = idx - start_day + 1
        if day_num < 1 or day_num > num_days:
            cell_html.append('<div class="calCell calEmpty"></div>')
            continue

        classes = ["calCell"]
        marker_tooltip = None

        # Priority: today ring > salary dot > holiday dot
        if today_day is not None and day_num == today_day:
            classes.append("calToday")
            marker_tooltip = "Today"
        elif use_marker_map and day_num in (salary_markers_by_day or {}):
            classes.append("calSalary")
            marker_tooltip = (salary_markers_by_day or {})[day_num]
        elif not use_marker_map and legacy_salary_day is not None and day_num == legacy_salary_day:
            classes.append("calSalary")
            marker_tooltip = "Salary day"
        elif day_num in holiday_days:
            classes.append("calHoliday")
            marker_tooltip = "Holiday"

        title_attr = f' title="{_escape_html(marker_tooltip)}"' if marker_tooltip else ""
        cell_html.append(
            f'<div class="{ " ".join(classes) }"{title_attr}><div class="dayNum">{day_num}</div></div>'
        )

    month_label = datetime(year, month, 1).strftime("%B %Y")

    html = f"""
<div class="calWrap" aria-label="Calendar {month_label}">
  <div class="calTitleRow">
    <div class="calTitle">{_escape_html(month_label)}</div>
    <div class="calLegend">
      <span class="legItem"><span class="legDot legHoliday"></span>Holiday</span>
      <span class="legItem"><span class="legDot legSalary"></span>Salary</span>
      <span class="legItem"><span class="legDot legToday"></span>Today</span>
    </div>
  </div>

  <div class="calGrid">
    <div class="calDow">Mon</div><div class="calDow">Tue</div><div class="calDow">Wed</div>
    <div class="calDow">Thu</div><div class="calDow">Fri</div><div class="calDow">Sat</div><div class="calDow">Sun</div>
    {''.join(cell_html)}
  </div>
</div>
<style>
  .calWrap {{
    --cal-bg: transparent;
    --cal-text: rgba(255,255,255,0.92);
    --cal-subtext: rgba(255,255,255,0.65);
    --cal-border: rgba(255,255,255,0.10);
    --cal-cell: rgba(255,255,255,0.06);
    --cal-cell-hover: rgba(255,255,255,0.10);
    --cal-empty: rgba(255,255,255,0.03);

    --holiday: rgba(255, 85, 85, 0.95);
    --salary: rgba(76, 201, 118, 0.95);
    --today: rgba(80, 170, 255, 0.95);
    color: var(--cal-text);
    background: var(--cal-bg);
    border: 1px solid var(--cal-border);
    border-radius: 14px;
    padding: 12px 12px 10px 12px;
  }}

  @media (prefers-color-scheme: light) {{
    .calWrap {{
      --cal-text: rgba(0,0,0,0.86);
      --cal-subtext: rgba(0,0,0,0.55);
      --cal-border: rgba(0,0,0,0.10);
      --cal-cell: rgba(0,0,0,0.05);
      --cal-cell-hover: rgba(0,0,0,0.08);
      --cal-empty: rgba(0,0,0,0.02);
    }}
  }}

  .calTitleRow {{
    display:flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
  }}
  .calTitle {{
    font-weight: 700;
    letter-spacing: 0.2px;
    font-size: 14px;
    color: var(--cal-text);
  }}
  .calLegend {{
    display:flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
  }}
  .legItem {{
    font-size: 12px;
    color: var(--cal-subtext);
    display:flex;
    align-items:center;
    gap: 6px;
  }}
  .legDot {{
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display:inline-block;
  }}
  .legHoliday {{ background: var(--holiday); }}
  .legSalary {{ background: var(--salary); }}
  .legToday {{ background: var(--today); }}

  .calGrid {{
    display:grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 6px;
  }}
  .calDow {{
    text-align:center;
    font-size: 11px;
    color: var(--cal-subtext);
    padding-bottom: 2px;
    font-weight: 600;
  }}
  .calCell {{
    height: 52px;
    border-radius: 12px;
    background: var(--cal-cell);
    border: 1px solid var(--cal-border);
    display:flex;
    align-items:flex-start;
    justify-content:flex-start;
    padding: 8px 8px;
    transition: background 120ms ease;
    position: relative;
    box-sizing: border-box;
  }}
  .calCell:hover {{
    background: var(--cal-cell-hover);
  }}
  .calEmpty {{
    height: 52px;
    background: var(--cal-empty);
    border: 1px dashed rgba(255,255,255,0.06);
  }}
  @media (prefers-color-scheme: light) {{
    .calEmpty {{
      border: 1px dashed rgba(0,0,0,0.06);
    }}
  }}
  .dayNum {{
    font-size: 12.5px;
    font-weight: 650;
    color: var(--cal-text);
  }}
  .calHoliday::after {{
    content: "";
    position:absolute;
    top: 10px;
    right: 10px;
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: var(--holiday);
    box-shadow: 0 0 0 3px rgba(255, 85, 85, 0.12);
  }}
  .calSalary::after {{
    content: "";
    position:absolute;
    top: 10px;
    right: 10px;
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: var(--salary);
    box-shadow: 0 0 0 3px rgba(76, 201, 118, 0.12);
  }}
  .calToday {{
    border-color: rgba(80, 170, 255, 0.45);
    background: rgba(80, 170, 255, 0.10);
  }}
  .calToday::after {{
    content: "";
    position:absolute;
    top: 9px;
    right: 9px;
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: transparent;
    border: 3px solid var(--today);
  }}
</style>
"""
    # Dedent to avoid markdown treating leading spaces as code blocks.
    return textwrap.dedent(html).strip()


