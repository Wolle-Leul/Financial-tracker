from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from sqlalchemy import func, select
import streamlit as st

from .auth import check_password
from .calendar_plot import generate_calendar_html
from .charts import make_sankey_income_expense
from .demo_data import get_demo_incomes
from .holidays import get_holidays_poland
from .metrics import compute_dashboard_metrics_from_db, compute_kpi_cards
from .salary import compute_salary_dates
from .services.import_service import import_statement_pdf
from .db.models import Category, Import as ImportModel, SubCategory, Transaction
from .db.seed import ensure_demo_categories_seeded
from .db.session import session_scope
from .db.user import get_or_create_default_user_id


def _format_amount(value: float) -> str:
    return "{:,.2f}".format(value)


@st.cache_data(ttl=300)
def _load_expenses_plan(user_id: int) -> pd.DataFrame:
    """
    Load the current dashboard expense-plan from DB:
    - Category / SubCategory (labels)
    - Amount / Deadline (planned defaults used until metrics are DB-driven)
    """
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

    # Coerce planned defaults into the schema the metrics module expects.
    if expenses.empty:
        return expenses

    expenses["Amount"] = expenses["Amount"].astype(float).fillna(0.0)
    expenses["Deadline"] = expenses["Deadline"].fillna(1).astype(int)
    return expenses


def render_app() -> None:
    if not check_password():
        return

    st.success("Welcome! You are logged in.")

    today = datetime.today()
    today_month = today.month
    today_year = today.year
    today_date = today.day

    # Holidays need to cover the previous and next year because salary
    # adjustments can cross year boundaries (e.g., Dec -> Jan).
    holiday_years = [today_year - 1, today_year, today_year + 1]
    holidays_all: List[str] = []
    for y in holiday_years:
        holidays_all.extend(get_holidays_poland(y))
    holidays_all = sorted(set(holidays_all))

    user_id = get_or_create_default_user_id()
    ensure_demo_categories_seeded(user_id)
    expenses_all = _load_expenses_plan(user_id)

    # Sidebar: month and year used for calendar only.
    month = st.sidebar.selectbox(
        "Select Month",
        range(1, 13),
        format_func=lambda x: datetime(1900, x, 1).strftime("%B"),
        index=today_month - 1,
    )
    year = st.sidebar.selectbox(
        "Select Year",
        range(today_year - 1, today_year + 1),
        index=1,
    )

    # Sidebar slicers for the "To Pay" table.
    st.sidebar.header("Category")
    selected_categories: List[str] = st.sidebar.multiselect(
        "Pick your Category",
        expenses_all["Category"].unique().tolist(),
    )

    if selected_categories:
        expenses_filtered = expenses_all[expenses_all["Category"].isin(selected_categories)]
    else:
        expenses_filtered = expenses_all.copy()

    st.sidebar.header("Sub-Category")
    selected_subcategories: List[str] = st.sidebar.multiselect(
        "Pick your Sub-Category",
        expenses_filtered["SubCategory"].unique().tolist(),
    )

    if selected_subcategories:
        expenses_filtered = expenses_filtered[expenses_filtered["SubCategory"].isin(selected_subcategories)]

    st.sidebar.header("PDF Import")
    uploaded_pdf = st.sidebar.file_uploader("Upload bank statement (PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        if st.sidebar.button("Import PDF"):
            with st.spinner("Parsing and saving transactions..."):
                try:
                    result = import_statement_pdf(uploaded_pdf.read(), uploaded_pdf.name)
                    st.session_state["last_import_id"] = result["import_id"]
                    st.sidebar.success(
                        f"Imported {result['rows_added']} transactions "
                        f"(import_id={result['import_id']}, parser_warnings={result.get('parse_warnings_count', 0)})"
                    )
                except Exception as e:
                    st.sidebar.error(f"Import failed: {e}")

    with st.sidebar.expander("Manual category override (uncategorized)", expanded=False):
        last_import_id = st.session_state.get("last_import_id")
        if not last_import_id:
            st.caption("Import a PDF to see uncategorized transactions.")
        else:
            with session_scope() as session:
                uncategorized_stmt = (
                    select(Transaction.id, Transaction.description)
                    .where(Transaction.user_id == user_id)
                    .where(Transaction.import_id == int(last_import_id))
                    .where(Transaction.subcategory_id.is_(None))
                    .limit(50)
                )
                uncategorized = session.execute(uncategorized_stmt).all()

                subcats_stmt = (
                    select(SubCategory.id, SubCategory.category_id, SubCategory.name, Category.name)
                    .join(Category, SubCategory.category_id == Category.id)
                    .where(Category.user_id == user_id)
                )
                subcats = session.execute(subcats_stmt).all()

            if not uncategorized:
                st.caption("All transactions have categories for this import.")
            else:
                txn_ids = [int(r[0]) for r in uncategorized]
                txn_labels = {int(r[0]): f"{int(r[0])}: {(r[1] or '')[:45]}" for r in uncategorized}
                subcat_by_id = {int(r[0]): (int(r[1]), f"{r[3]} - {r[2]}") for r in subcats}
                subcat_ids = list(subcat_by_id.keys())

                picked_txn_id = st.selectbox(
                    "Transaction",
                    options=txn_ids,
                    format_func=lambda tid: txn_labels.get(tid, str(tid)),
                    key="picked_txn_id",
                )
                picked_subcat_id = st.selectbox(
                    "Map to subcategory",
                    options=subcat_ids,
                    format_func=lambda sid: subcat_by_id.get(sid, (None, str(sid)))[1],
                    key="picked_subcat_id",
                )

                if st.button("Save mapping", key="save_mapping"):
                    with session_scope() as session:
                        # Ensure we only update rows for the current user.
                        txn = session.execute(
                            select(Transaction).where(
                                Transaction.user_id == user_id, Transaction.id == int(picked_txn_id)
                            )
                        ).scalar_one()
                        category_id, _label = subcat_by_id[int(picked_subcat_id)]
                        txn.category_id = category_id
                        txn.subcategory_id = int(picked_subcat_id)

                    st.success("Mapping saved.")
                    st.rerun()

    # Metrics are computed for the selected month/year (based on a reference day in that month).
    ref_day = min(today.day, calendar.monthrange(year, month)[1])
    reference_date = datetime(year, month, ref_day)

    salary_prev_month, due_salary_date = compute_salary_dates(reference_date, holidays_all)

    # Query imported transactions within the current salary window.
    selected_subcat_names = set(expenses_filtered["SubCategory"].tolist())
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
                expense_amounts_by_subcategory_all[subcat_str] = expense_amounts_by_subcategory_all.get(subcat_str, 0.0) + abs_amt

            if subcat_str in selected_subcat_names:
                expense_amounts_by_subcategory[subcat_str] = expense_amounts_by_subcategory.get(subcat_str, 0.0) + abs_amt

    metrics, to_pay = compute_dashboard_metrics_from_db(
        expenses_df_filtered=expenses_filtered,
        salary_prev_month=salary_prev_month,
        due_salary_date=due_salary_date,
        reference_date=reference_date,
        income_total=income_total,
        expense_total=expense_total,
        groceries_total=groceries_total,
        expense_amounts_by_subcategory=expense_amounts_by_subcategory,
    )

    # -------------------------
    # Important & Interesting KPIs
    # -------------------------
    cash_left = metrics.net_of_net

    # Budget KPIs (based on selected filters).
    planned_selected_total = float(expenses_filtered["Amount"].astype(float).sum()) if not expenses_filtered.empty else 0.0
    actual_selected_total = float(sum(expense_amounts_by_subcategory.values())) if expense_amounts_by_subcategory else 0.0
    budget_variance = actual_selected_total - planned_selected_total
    budget_remaining = planned_selected_total - actual_selected_total

    # Upcoming bills within next 30 days (based on selected filters).
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

    # Forecast end-of-window net based on current burn pace.
    total_window_days = metrics.days_from_prev_salary + metrics.days_till_next_salary
    forecast_net = cash_left
    if metrics.days_from_prev_salary > 0 and total_window_days > 0:
        forecast_expenses = expense_total * (total_window_days / metrics.days_from_prev_salary)
        forecast_net = income_total - forecast_expenses

    # Top spending category in the salary window.
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

    # Import quality (latest import row).
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
    )

    kpis = [(c.title, c.value, c.subtitle) for c in kpi_cards]

    st.markdown("### Important & Interesting")
    kpi_rows = [kpis[0:6], kpis[6:12]]
    for row_idx, row_kpis in enumerate(kpi_rows):
        cols = st.columns(6)
        for col, (title, value, subtitle) in zip(cols, row_kpis):
            with col:
                # st.metric gives a polished card-like UI automatically.
                st.metric(label=title, value=value)
                st.caption(str(subtitle))

    incomes_df = pd.DataFrame({"Source": ["Income"], "Amount": [income_total]})
    expenses_sankey_df = pd.DataFrame(
        {"SubCategory": list(expense_amounts_by_subcategory_all.keys()), "Amount": list(expense_amounts_by_subcategory_all.values())}
    )

    # UI layout: 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.text(f"{metrics.days_till_next_salary} days left for infy")
        with st.expander("Calendar", expanded=False):
            st.subheader("📅 Calendar")
            st.markdown(
                generate_calendar_html(year=year, month=month, holidays=holidays_all, today=today),
                unsafe_allow_html=True,
            )

    with col2:
        due_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">Due date </p>'
        st.markdown(due_title, unsafe_allow_html=True)
        st.text(str(metrics.due_salary_date.strftime("%Y-%m-%d")))

        groceries_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">Groceries </p>'
        st.markdown(groceries_title, unsafe_allow_html=True)
        st.text(str(f"{round(metrics.groceries_amount, 2):,}") + " $ ")

        net_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 15px;">Net of Net </p>'
        st.markdown(net_title, unsafe_allow_html=True)
        st.text(str(f"{round(metrics.net_of_net, 2):,}") + " $")

        needs_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">Need%</p>'
        st.markdown(needs_title, unsafe_allow_html=True)
        st.text(str(f"{round(metrics.needs_percent, 2):,}") + " %")

        delta_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">ΔTarget%</p>'
        st.markdown(delta_title, unsafe_allow_html=True)
        st.text(str(f"{round(metrics.target_vs_actual_percent, 2):,}") + " %")

    with col3:
        fig = make_sankey_income_expense(incomes_df=incomes_df, expenses_df_filtered=expenses_sankey_df)
        st.plotly_chart(fig, use_container_width=True, unsafe_allow_html=True)

    # To Pay table: keep it where the original app rendered it (col2 area).
    # This is a minor layout shift, but keeps the primary information.
    with st.expander("To Pay", expanded=False):
        if to_pay.empty:
            st.caption("No categories configured yet. Import a PDF to seed transactions and mapping.")
        elif float(to_pay["ActualAmount"].sum()) == 0.0:
            st.caption("No expenses found in this salary window (for the selected category filters).")
        else:
            to_pay_display = to_pay.copy()
            # Keep numeric columns numeric for sorting; round for readability.
            to_pay_display["PlannedAmount"] = to_pay_display["PlannedAmount"].astype(float).round(2)
            to_pay_display["ActualAmount"] = to_pay_display["ActualAmount"].astype(float).round(2)
            to_pay_display["Variance"] = to_pay_display["Variance"].astype(float).round(2)

            st.caption("Variance = ActualAmount - PlannedAmount (positive means over budget).")
            st.dataframe(to_pay_display, use_container_width=True, hide_index=True)

