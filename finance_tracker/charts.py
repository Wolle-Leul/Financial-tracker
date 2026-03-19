from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go


def make_sankey_income_expense(
    incomes_df: pd.DataFrame,
    expenses_df_filtered: pd.DataFrame,
    *,
    currency: str = "PLN",
    top_n_expenses: int = 6,
    other_label: str = "Other",
) -> go.Figure:
    """
    Create a Sankey diagram showing:
    Income -> Total -> Expenses -> Left

    UX upgrades:
    - stable ordering (desc by amount for expenses)
    - top-N expenses, with an "Other" bucket
    - currency-annotated node labels for clarity
    """
    if incomes_df is None or incomes_df.empty:
        incomes_df = pd.DataFrame({"Source": [], "Amount": []})
    if expenses_df_filtered is None or expenses_df_filtered.empty:
        expenses_df_filtered = pd.DataFrame({"SubCategory": [], "Amount": []})

    inc_df = incomes_df.copy()
    inc_df["Amount"] = inc_df["Amount"].astype(float)

    exp_df = expenses_df_filtered.copy()
    exp_df["Amount"] = exp_df["Amount"].astype(float)
    exp_df = exp_df[exp_df["Amount"] > 0]

    exp_df = exp_df.sort_values("Amount", ascending=False)

    top_exp = exp_df.head(top_n_expenses)
    other_amount = float(exp_df["Amount"].iloc[top_n_expenses:].sum()) if len(exp_df) > top_n_expenses else 0.0
    if other_amount > 0:
        top_exp = pd.concat(
            [
                top_exp,
                pd.DataFrame({"SubCategory": [other_label], "Amount": [other_amount]}),
            ],
            ignore_index=True,
        )

    inc_labels: List[str] = inc_df["Source"].astype(str).tolist()
    inc_amounts: List[float] = inc_df["Amount"].tolist()

    exp_labels: List[str] = top_exp["SubCategory"].astype(str).tolist()
    exp_amounts: List[float] = top_exp["Amount"].tolist()

    total_income = sum(inc_amounts)
    total_expenses = sum(exp_amounts)
    left = total_income - total_expenses

    inc_labels_annotated = [f"{lbl} ({amt:,.0f} {currency})" for lbl, amt in zip(inc_labels, inc_amounts)]
    exp_labels_annotated = [f"{lbl} ({amt:,.0f} {currency})" for lbl, amt in zip(exp_labels, exp_amounts)]

    source = inc_labels_annotated + ["Total"] + exp_labels_annotated + [f"Left ({left:,.0f} {currency})"]

    inc_amounts_with_total_zero = inc_amounts + [0.0]
    exp_amounts_with_left = exp_amounts + [left]
    value = inc_amounts_with_total_zero + exp_amounts_with_left

    # Build edges around the "Total" node.
    center = source.index("Total")
    source_indexes: List[int] = []
    target_indexes: List[int] = []
    for i in range(len(source)):
        if i <= center:
            source_indexes.append(i)
        else:
            source_indexes.append(center)

    for i in range(len(source)):
        if i < center:
            target_indexes.append(center)
        else:
            target_indexes.append(i)

    link = dict(source=source_indexes, target=target_indexes, value=value)
    node = dict(label=source, pad=16, thickness=26, color="#89CFF0")
    fig = go.Figure(go.Sankey(link=link, node=node, orientation="v"))
    fig.update_layout(margin=dict(l=0, r=0, t=5, b=5), width=600)
    return fig

