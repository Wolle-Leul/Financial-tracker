from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    password: str = Field(..., min_length=1)


class KpiItem(BaseModel):
    title: str
    value: str
    subtitle: str
    kind: Optional[str] = "default"


class ToPayRow(BaseModel):
    Category: str
    SubCategory: str
    Deadline: Optional[str] = None
    PlannedAmount: float
    ActualAmount: float
    Variance: float


class DashboardResponse(BaseModel):
    kpis: List[KpiItem]
    filter_categories: List[str] = Field(default_factory=list)
    filter_subcategories: List[str] = Field(default_factory=list)
    days_till_next_salary: int
    days_left_for_infy_label: str
    due_salary_date: str
    groceries_amount: float
    net_of_net: float
    needs_percent: float
    target_vs_actual_percent: float
    salary_prev_month: str
    calendar_year: int
    calendar_month: int
    calendar_html: str
    sankey_plotly: dict[str, Any]
    to_pay_rows: List[ToPayRow]
    to_pay_empty_reason: Optional[str] = None
    import_quality_value: str
    import_quality_sub: str
    expected_income_net: Optional[float] = None
    income_variance_vs_expected_percent: Optional[float] = None


class UncategorizedItem(BaseModel):
    id: int
    label: str


class SubcategoryOption(BaseModel):
    id: int
    label: str


class UncategorizedResponse(BaseModel):
    transactions: List[UncategorizedItem]
    subcategories: List[SubcategoryOption]


class MapTransactionRequest(BaseModel):
    subcategory_id: int


class ImportPdfResponse(BaseModel):
    import_id: int
    rows_added: int
    parsed_total: int
    parse_warnings_count: int
    categorized_txn_count: int
    uncategorized_txn_count: int
