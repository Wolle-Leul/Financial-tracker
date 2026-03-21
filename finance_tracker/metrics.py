from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class DashboardMetrics:
    days_till_next_salary: int
    days_from_prev_salary: int
    groceries_amount: float
    net_of_net: float
    needs_percent: float
    target_vs_actual_percent: float
    due_salary_date: datetime
    salary_prev_month: datetime


@dataclass(frozen=True)
class KpiCard:
    title: str
    value: str
    subtitle: str


def resolve_budget_target_ratio(strategy: str, stored: float) -> float:
    """
    Map a named budgeting strategy to an effective target ratio used in metrics.
    `stored` is the user's configured target_ratio (e.g. 0.45).
    """
    s = (strategy or "custom_target_ratio").lower().replace("-", "_")
    if s == "classic_50_30_20":
        return 0.50
    if s == "zero_based":
        return max(min(float(stored), 0.99), 0.05)
    if s == "salary_window_only":
        return float(stored)
    return float(stored)


def _compute_groceries_allocation(
    expenses_df: pd.DataFrame,
    days_till_next_salary: int,
    days_from_prev_salary: int,
) -> float:
    groceries_rows = expenses_df[expenses_df["SubCategory"].str.contains("Groceries", na=False)]
    gr_total = groceries_rows["Amount"].astype(float).sum()
    denom = days_till_next_salary + days_from_prev_salary
    if denom <= 0:
        return 0.0
    return (gr_total / denom) * days_till_next_salary


def compute_dashboard_metrics(
    expenses_df_filtered: pd.DataFrame,
    expenses_df_all: pd.DataFrame,
    incomes_df: pd.DataFrame,
    salary_prev_month: datetime,
    due_salary_date: datetime,
    today: datetime,
    target_ratio: float = 0.45,
) -> Tuple[DashboardMetrics, pd.DataFrame]:
    """
    Compute metrics and the "To Pay" table based on selected filters.
    """
    days_till_next_salary = (due_salary_date - today).days
    days_from_prev_salary = (today - salary_prev_month).days

    groceries_amount = _compute_groceries_allocation(expenses_df_all, days_till_next_salary, days_from_prev_salary)

    # net-of-net: income - expenses (sum of selected/all expenses)
    total_income = incomes_df["Amount"].astype(float).sum()
    total_expenses = expenses_df_all["Amount"].astype(float).sum()
    net_of_net = total_income - total_expenses

    # target
    # In the original app: Target_incom = (expenses_sum / 0.45)
    # and TargetVsActual uses net_of_net / 0.45.
    if total_income <= 0:
        needs_percent = 0.0
    else:
        needs_percent = (total_expenses / total_income) * 100

    target_net_of_net = net_of_net / target_ratio if target_ratio else 0.0
    if target_net_of_net == 0:
        target_vs_actual_percent = 0.0
    else:
        target_vs_actual_percent = ((target_net_of_net - net_of_net) / target_net_of_net) * 100

    # Build To Pay table: show selected expenses with groceries deadline overridden.
    to_pay = expenses_df_filtered.copy()

    # Deadline becomes a concrete date. Original logic:
    # - For groceries: set to due_salary_date
    # - For others: set month/year to salary_prev_month; keep day-of-month from Deadline column
    def deadline_for_row(row) -> date:
        sub_cat = str(row.get("SubCategory", ""))
        if "Groceries" in sub_cat:
            return due_salary_date.date()

        # Deadline is originally a day-of-month integer.
        try:
            dom = int(row["Deadline"])
        except Exception:
            dom = salary_prev_month.day
        return date(salary_prev_month.year, salary_prev_month.month, dom)

    to_pay["Deadline"] = to_pay.apply(deadline_for_row, axis=1)

    # Replace groceries amount with computed groceries allocation
    groceries_mask = to_pay["SubCategory"].str.contains("Groceries", na=False)
    to_pay.loc[groceries_mask, "Amount"] = groceries_amount

    metrics = DashboardMetrics(
        days_till_next_salary=days_till_next_salary,
        days_from_prev_salary=days_from_prev_salary,
        groceries_amount=groceries_amount,
        net_of_net=net_of_net,
        needs_percent=needs_percent,
        target_vs_actual_percent=target_vs_actual_percent,
        due_salary_date=due_salary_date,
        salary_prev_month=salary_prev_month,
    )
    return metrics, to_pay


def compute_dashboard_metrics_from_db(
    expenses_df_filtered: pd.DataFrame,
    salary_prev_month: datetime,
    due_salary_date: datetime,
    reference_date: datetime,
    income_total: float,
    expense_total: float,
    groceries_total: float,
    expense_amounts_by_subcategory: dict[str, float],
    target_ratio: float = 0.45,
    budget_strategy: str = "custom_target_ratio",
) -> Tuple[DashboardMetrics, pd.DataFrame]:
    """
    Compute dashboard metrics using imported transactions.

    - `income_total` / `expense_total` are sums for the salary window
    - `groceries_total` is the groceries subset of expenses
    - `expense_amounts_by_subcategory` drives the "To Pay" table
    """
    eff_ratio = resolve_budget_target_ratio(budget_strategy, target_ratio)

    days_till_next_salary = (due_salary_date - reference_date).days
    days_from_prev_salary = (reference_date - salary_prev_month).days

    net_of_net = income_total - expense_total

    if income_total <= 0:
        needs_percent = 0.0
    else:
        needs_percent = (expense_total / income_total) * 100

    target_net_of_net = net_of_net / eff_ratio if eff_ratio else 0.0
    if target_net_of_net == 0:
        target_vs_actual_percent = 0.0
    else:
        target_vs_actual_percent = ((target_net_of_net - net_of_net) / target_net_of_net) * 100

    groceries_amount = float(groceries_total)

    # Build To Pay table from selected subcategories (labels + planned deadline day).
    # We keep planned amounts, add actual amounts, and compute variance.
    to_pay = expenses_df_filtered.copy()

    def deadline_for_row(row) -> date:
        sub_cat = str(row.get("SubCategory", ""))
        if "Groceries" in sub_cat:
            return due_salary_date.date()
        dom = row.get("Deadline")
        try:
            dom_int = int(dom)
        except Exception:
            dom_int = salary_prev_month.day
        return date(salary_prev_month.year, salary_prev_month.month, dom_int)

    to_pay["Deadline"] = to_pay.apply(deadline_for_row, axis=1)

    # Keep planned budgets, then attach actuals from imported transactions.
    to_pay["PlannedAmount"] = to_pay["Amount"].astype(float)
    def amount_for_row(row) -> float:
        sub_cat = str(row.get("SubCategory", ""))
        return float(expense_amounts_by_subcategory.get(sub_cat, 0.0))

    to_pay["ActualAmount"] = to_pay.apply(amount_for_row, axis=1)
    to_pay["Variance"] = to_pay["ActualAmount"] - to_pay["PlannedAmount"]

    # Drop the original planned "Amount" column to avoid confusion.
    to_pay = to_pay.drop(columns=["Amount"])

    metrics = DashboardMetrics(
        days_till_next_salary=int(days_till_next_salary),
        days_from_prev_salary=int(days_from_prev_salary),
        groceries_amount=groceries_amount,
        net_of_net=float(net_of_net),
        needs_percent=float(needs_percent),
        target_vs_actual_percent=float(target_vs_actual_percent),
        due_salary_date=due_salary_date,
        salary_prev_month=salary_prev_month,
    )

    return metrics, to_pay


def compute_kpi_cards(
    *,
    cash_left: float,
    income_total: float,
    expense_total: float,
    groceries_total: float,
    days_from_prev_salary: int,
    days_till_next_salary: int,
    target_vs_actual_percent: float,
    budget_variance: float,
    budget_remaining: float,
    upcoming_bills_count: int,
    forecast_net: float,
    import_quality_value: str,
    import_quality_sub: str,
    top_category_name: str,
    top_category_amount: float,
    effective_target_ratio: float = 0.45,
    budget_strategy: str = "custom_target_ratio",
) -> List[KpiCard]:
    """
    Compute the KPI card payload used by the UI.

    This centralizes formatting + divide-by-zero guardrails.
    """

    income_total_safe = income_total if income_total > 0 else 0.0
    savings_rate = (cash_left / income_total_safe) * 100 if income_total_safe > 0 else 0.0
    groceries_share = (groceries_total / expense_total) * 100 if expense_total > 0 else 0.0
    burn_rate = (expense_total / days_from_prev_salary) if days_from_prev_salary > 0 else 0.0

    def _signed_pln(val: float) -> str:
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:,.0f} PLN"

    kpis = [
        KpiCard("Cash left until next salary", f"{cash_left:,.0f} PLN", "Income - spending in salary window"),
        KpiCard("Savings rate", f"{savings_rate:.1f}%", "Cash left / income"),
        KpiCard(
            "Daily spend (burn rate)",
            f"{burn_rate:,.0f} PLN/day" if days_from_prev_salary > 0 else "—",
            "Expenses pace",
        ),
        KpiCard("Groceries share of spending", f"{groceries_share:.1f}%", "Groceries / total expenses"),
        KpiCard(
            "Target progress (ΔTarget%)",
            f"{target_vs_actual_percent:.1f}%",
            f"Vs {effective_target_ratio * 100:.0f}% ratio ({budget_strategy})",
        ),
        KpiCard("Budget variance (selected)", _signed_pln(budget_variance), "Actual - planned"),
        KpiCard("Budget remaining (selected)", _signed_pln(budget_remaining), "Planned - actual"),
        KpiCard("Upcoming bills (30d)", f"{upcoming_bills_count}", "Based on planned deadlines"),
        KpiCard("Salary timing", f"{days_till_next_salary}d", f"{days_from_prev_salary}d since prev salary"),
        KpiCard("Forecast end-of-window net", f"{forecast_net:,.0f} PLN", "Based on current burn rate"),
        KpiCard("Import quality", import_quality_value, import_quality_sub),
        KpiCard("Top spending category", f"{top_category_name}", f"{top_category_amount:,.0f} PLN in window"),
    ]

    return kpis

