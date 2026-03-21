from __future__ import annotations

import calendar
import json
from datetime import datetime, timedelta
from typing import List, Optional, Set

import pandas as pd
from sqlalchemy import func, select

from finance_tracker.calendar_plot import generate_calendar_html
from finance_tracker.charts import make_sankey_income_expense
from finance_tracker.config import get_config
from finance_tracker.db.models import Category, Import as ImportModel, SubCategory, Transaction
from finance_tracker.db.salary_rule import get_or_create_salary_rule
from finance_tracker.db.session import session_scope
from finance_tracker.holidays import get_holidays_poland
from finance_tracker.metrics import (
    compute_dashboard_metrics_from_db,
    compute_kpi_cards,
    resolve_budget_target_ratio,
)
from finance_tracker.services.income_expectations import sum_expected_net_monthly
from finance_tracker.salary import compute_salary_dates
from finance_tracker.schemas.dashboard import DashboardResponse, KpiItem, ToPayRow


def load_expenses_plan(user_id: int) -> pd.DataFrame:
    """Load category/subcategory planned amounts from the DB (same as Streamlit `_load_expenses_plan`)."""
    with session_scope() as session:
        stmt = (
            select(
                Category.name,
                SubCategory.name,
                SubCategory.planned_amount,
                SubCategory.planned_deadline_day,
            )
            .join(SubCategory, SubCategory.category_id == Category.id)
            .where(Category.user_id == user_id)
        )
        rows = session.execute(stmt).all()

    expenses = pd.DataFrame(
        rows,
        columns=["Category", "SubCategory", "Amount", "Deadline"],
    )

    if expenses.empty:
        return expenses

    expenses["Amount"] = expenses["Amount"].astype(float).fillna(0.0)
    expenses["Deadline"] = expenses["Deadline"].fillna(1).astype(int)
    return expenses


def build_dashboard_response(
    user_id: int,
    *,
    year: int,
    month: int,
    selected_categories: Optional[List[str]] = None,
    selected_subcategories: Optional[List[str]] = None,
) -> DashboardResponse:
    """
    Pure domain orchestration for the dashboard (no Streamlit).
    Mirrors `render_app()` metrics in `finance_tracker/ui.py`.
    """
    selected_categories = selected_categories or []
    selected_subcategories = selected_subcategories or []

    today = datetime.today()
    today_year = today.year

    holiday_years = [today_year - 1, today_year, today_year + 1]
    holidays_all: List[str] = []
    for y in holiday_years:
        holidays_all.extend(get_holidays_poland(y))
    holidays_all = sorted(set(holidays_all))

    expenses_all = load_expenses_plan(user_id)

    filter_categories = sorted(expenses_all["Category"].dropna().unique().tolist()) if not expenses_all.empty else []

    if selected_categories:
        expenses_for_subcat_list = expenses_all[expenses_all["Category"].isin(selected_categories)]
    else:
        expenses_for_subcat_list = expenses_all.copy()

    filter_subcategories = (
        sorted(expenses_for_subcat_list["SubCategory"].dropna().unique().tolist())
        if not expenses_for_subcat_list.empty
        else []
    )

    if selected_categories:
        expenses_filtered = expenses_all[expenses_all["Category"].isin(selected_categories)]
    else:
        expenses_filtered = expenses_all.copy()

    if selected_subcategories:
        expenses_filtered = expenses_filtered[expenses_filtered["SubCategory"].isin(selected_subcategories)]

    ref_day = min(today.day, calendar.monthrange(year, month)[1])
    reference_date = datetime(year, month, ref_day)

    salary_rule = get_or_create_salary_rule(user_id)
    salary_dom = int(salary_rule.salary_day_of_month)
    salary_prev_month, due_salary_date = compute_salary_dates(
        reference_date,
        holidays_all,
        salary_day_of_month=salary_dom,
    )
    cfg = get_config()
    target_ratio = float(salary_rule.target_ratio) if salary_rule.target_ratio is not None else cfg.target_ratio

    selected_subcat_names: Set[str] = set(expenses_filtered["SubCategory"].tolist())
    expense_amounts_by_subcategory: dict[str, float] = {}
    expense_amounts_by_subcategory_all: dict[str, float] = {}

    income_total = 0.0
    expense_total = 0.0
    groceries_total = 0.0

    with session_scope() as session:
        txn_stmt = (
            select(Transaction.amount, SubCategory.name)
            .select_from(Transaction)
            .join(SubCategory, Transaction.subcategory_id == SubCategory.id, isouter=True)
            .where(Transaction.user_id == user_id)
            .where(Transaction.txn_datetime >= salary_prev_month)
            .where(Transaction.txn_datetime <= due_salary_date)
        )
        rows = session.execute(txn_stmt).all()

    for amount_dec, subcat_name in rows:
        amount = float(amount_dec)
        if amount > 0:
            income_total += amount
            continue
        if amount < 0:
            abs_amt = abs(amount)
            expense_total += abs_amt

            subcat_str = str(subcat_name) if subcat_name is not None else ""
            if "Groceries" in subcat_str:
                groceries_total += abs_amt

            if subcat_str:
                expense_amounts_by_subcategory_all[subcat_str] = (
                    expense_amounts_by_subcategory_all.get(subcat_str, 0.0) + abs_amt
                )

            if subcat_str in selected_subcat_names:
                expense_amounts_by_subcategory[subcat_str] = (
                    expense_amounts_by_subcategory.get(subcat_str, 0.0) + abs_amt
                )

    budget_strategy = str(salary_rule.budget_strategy)
    metrics, to_pay = compute_dashboard_metrics_from_db(
        expenses_df_filtered=expenses_filtered,
        salary_prev_month=salary_prev_month,
        due_salary_date=due_salary_date,
        reference_date=reference_date,
        income_total=income_total,
        expense_total=expense_total,
        groceries_total=groceries_total,
        expense_amounts_by_subcategory=expense_amounts_by_subcategory,
        target_ratio=target_ratio,
        budget_strategy=budget_strategy,
    )

    expected_income_net: Optional[float] = None
    income_variance_vs_expected_percent: Optional[float] = None
    try:
        exp_sum = sum_expected_net_monthly(user_id)
        if exp_sum > 0:
            expected_income_net = round(exp_sum, 2)
            income_variance_vs_expected_percent = round(
                ((income_total - exp_sum) / exp_sum) * 100.0,
                1,
            )
    except Exception:
        pass

    cash_left = metrics.net_of_net

    planned_selected_total = (
        float(expenses_filtered["Amount"].astype(float).sum()) if not expenses_filtered.empty else 0.0
    )
    actual_selected_total = (
        float(sum(expense_amounts_by_subcategory.values())) if expense_amounts_by_subcategory else 0.0
    )
    budget_variance = actual_selected_total - planned_selected_total
    budget_remaining = planned_selected_total - actual_selected_total

    window_start = reference_date.date()
    window_end = (reference_date + timedelta(days=30)).date()
    upcoming_bills_count = 0
    if not expenses_filtered.empty:
        for _, row in expenses_filtered.iterrows():
            planned_amt = float(row.get("Amount") or 0.0)
            if planned_amt <= 0:
                continue

            subcat = str(row.get("SubCategory") or "")
            if "Groceries" in subcat:
                deadline_dt = metrics.due_salary_date.date()
            else:
                try:
                    dom_int = int(row.get("Deadline"))
                except Exception:
                    dom_int = metrics.salary_prev_month.day
                deadline_dt = datetime(
                    metrics.salary_prev_month.year,
                    metrics.salary_prev_month.month,
                    dom_int,
                ).date()

            if window_start <= deadline_dt <= window_end:
                upcoming_bills_count += 1

    total_window_days = metrics.days_from_prev_salary + metrics.days_till_next_salary
    forecast_net = cash_left
    if metrics.days_from_prev_salary > 0 and total_window_days > 0:
        forecast_expenses = expense_total * (total_window_days / metrics.days_from_prev_salary)
        forecast_net = income_total - forecast_expenses

    top_category_name = "—"
    top_category_amount = 0.0
    with session_scope() as session:
        cat_stmt = (
            select(Category.name, func.sum(func.abs(Transaction.amount)).label("expense_total"))
            .select_from(Transaction)
            .join(SubCategory, Transaction.subcategory_id == SubCategory.id, isouter=True)
            .join(Category, SubCategory.category_id == Category.id, isouter=True)
            .where(Transaction.user_id == user_id)
            .where(Transaction.txn_datetime >= salary_prev_month)
            .where(Transaction.txn_datetime <= due_salary_date)
            .where(Transaction.amount < 0)
            .group_by(Category.name)
            .order_by(func.sum(func.abs(Transaction.amount)).desc())
            .limit(1)
        )
        row = session.execute(cat_stmt).first()
        if row and row[0] is not None:
            top_category_name = str(row[0])
            top_category_amount = float(row[1] or 0.0)

    import_quality_value = "—"
    import_quality_sub = "No imports yet"
    with session_scope() as session:
        latest_import_stmt = (
            select(ImportModel)
            .where(ImportModel.user_id == user_id)
            .order_by(ImportModel.imported_at.desc())
            .limit(1)
        )
        latest_import = session.execute(latest_import_stmt).scalar_one_or_none()
        if latest_import is not None and latest_import.rows_parsed > 0:
            categorized = int(latest_import.categorized_txn_count or 0)
            total = int(latest_import.rows_parsed or 0)
            ratio = categorized / total if total > 0 else 0.0
            import_quality_value = f"{ratio * 100:.0f}%"
            import_quality_sub = f"{int(latest_import.parse_warnings_count or 0)} warnings"
        elif latest_import is not None:
            import_quality_value = "0%"
            import_quality_sub = f"{int(latest_import.parse_warnings_count or 0)} warnings"

    eff_tr = resolve_budget_target_ratio(budget_strategy, target_ratio)
    kpi_cards = compute_kpi_cards(
        cash_left=cash_left,
        income_total=income_total,
        expense_total=expense_total,
        groceries_total=groceries_total,
        days_from_prev_salary=metrics.days_from_prev_salary,
        days_till_next_salary=metrics.days_till_next_salary,
        target_vs_actual_percent=metrics.target_vs_actual_percent,
        budget_variance=budget_variance,
        budget_remaining=budget_remaining,
        upcoming_bills_count=upcoming_bills_count,
        forecast_net=forecast_net,
        import_quality_value=import_quality_value,
        import_quality_sub=import_quality_sub,
        top_category_name=top_category_name,
        top_category_amount=top_category_amount,
        effective_target_ratio=eff_tr,
        budget_strategy=budget_strategy,
    )

    kpis = [KpiItem(title=c.title, value=c.value, subtitle=c.subtitle) for c in kpi_cards]

    incomes_df = pd.DataFrame({"Source": ["Income"], "Amount": [income_total]})
    expenses_sankey_df = pd.DataFrame(
        {
            "SubCategory": list(expense_amounts_by_subcategory_all.keys()),
            "Amount": list(expense_amounts_by_subcategory_all.values()),
        }
    )

    fig = make_sankey_income_expense(incomes_df=incomes_df, expenses_df_filtered=expenses_sankey_df)
    sankey_plotly = json.loads(fig.to_json())

    calendar_html = generate_calendar_html(
        year=year,
        month=month,
        holidays=holidays_all,
        today=today,
        salary_date=salary_dom,
    )

    to_pay_empty_reason: Optional[str] = None
    to_pay_rows: List[ToPayRow] = []
    if to_pay.empty:
        to_pay_empty_reason = "No categories configured yet. Import a PDF to seed transactions and mapping."
    elif float(to_pay["ActualAmount"].sum()) == 0.0:
        to_pay_empty_reason = "No expenses found in this salary window (for the selected category filters)."
    else:
        to_pay_display = to_pay.copy()
        to_pay_display["PlannedAmount"] = to_pay_display["PlannedAmount"].astype(float).round(2)
        to_pay_display["ActualAmount"] = to_pay_display["ActualAmount"].astype(float).round(2)
        to_pay_display["Variance"] = to_pay_display["Variance"].astype(float).round(2)

        for _, r in to_pay_display.iterrows():
            dl = r.get("Deadline")
            dl_str: Optional[str]
            if dl is None:
                dl_str = None
            elif hasattr(dl, "isoformat"):
                dl_str = dl.isoformat()
            else:
                dl_str = str(dl)

            to_pay_rows.append(
                ToPayRow(
                    Category=str(r.get("Category", "")),
                    SubCategory=str(r.get("SubCategory", "")),
                    Deadline=dl_str,
                    PlannedAmount=float(r.get("PlannedAmount", 0.0)),
                    ActualAmount=float(r.get("ActualAmount", 0.0)),
                    Variance=float(r.get("Variance", 0.0)),
                )
            )

    return DashboardResponse(
        kpis=kpis,
        filter_categories=filter_categories,
        filter_subcategories=filter_subcategories,
        days_till_next_salary=metrics.days_till_next_salary,
        days_left_for_infy_label=f"{metrics.days_till_next_salary} days left for infy",
        due_salary_date=metrics.due_salary_date.strftime("%Y-%m-%d"),
        groceries_amount=float(metrics.groceries_amount),
        net_of_net=float(metrics.net_of_net),
        needs_percent=float(metrics.needs_percent),
        target_vs_actual_percent=float(metrics.target_vs_actual_percent),
        salary_prev_month=metrics.salary_prev_month.isoformat(),
        calendar_year=year,
        calendar_month=month,
        calendar_html=calendar_html,
        sankey_plotly=sankey_plotly,
        to_pay_rows=to_pay_rows,
        to_pay_empty_reason=to_pay_empty_reason,
        import_quality_value=import_quality_value,
        import_quality_sub=import_quality_sub,
        expected_income_net=expected_income_net,
        income_variance_vs_expected_percent=income_variance_vs_expected_percent,
    )
