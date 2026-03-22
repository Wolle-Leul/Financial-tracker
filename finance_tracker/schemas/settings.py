from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field


class SubcategoryCreatedOut(BaseModel):
    id: int
    category_id: int
    name: str


class BudgetCategoryOut(BaseModel):
    id: int
    name: str
    subcategories: List[SubcategoryCreatedOut] = Field(default_factory=list)


class BudgetLabelsResponse(BaseModel):
    categories: List[BudgetCategoryOut] = Field(default_factory=list)


class SubcategoryCreateBody(BaseModel):
    category_id: int = Field(..., gt=0)
    name: str = Field(..., min_length=1, max_length=120)
    match_keywords: Optional[str] = Field(None, max_length=2000)


class SalaryRuleResponse(BaseModel):
    salary_day_of_month: int = Field(..., ge=1, le=31)
    holiday_country: str
    target_ratio: float = Field(..., gt=0, lt=1)
    budget_strategy: str


class SalaryRulePatch(BaseModel):
    salary_day_of_month: Optional[int] = Field(None, ge=1, le=31)
    holiday_country: Optional[str] = Field(None, max_length=60)
    target_ratio: Optional[float] = Field(None, gt=0, lt=1)
    budget_strategy: Optional[str] = Field(None, max_length=64)


class IncomeSourceItem(BaseModel):
    id: int
    label: str
    employer_name: Optional[str] = None
    contract_type: str
    gross_amount: Optional[float] = None
    net_amount: Optional[float] = None
    use_net_only: bool
    sort_order: int


class IncomeSourceCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)
    employer_name: Optional[str] = Field(None, max_length=255)
    contract_type: str = Field("other", max_length=64)
    gross_amount: Optional[Decimal] = None
    net_amount: Optional[Decimal] = None
    use_net_only: bool = True
    sort_order: int = 0


class IncomeSourcePatch(BaseModel):
    label: Optional[str] = Field(None, max_length=120)
    employer_name: Optional[str] = None
    contract_type: Optional[str] = Field(None, max_length=64)
    gross_amount: Optional[Decimal] = None
    net_amount: Optional[Decimal] = None
    use_net_only: Optional[bool] = None
    sort_order: Optional[int] = None


class SubcategoryRecurringItem(BaseModel):
    id: int
    category_name: str
    name: str
    planned_amount: Optional[float] = None
    planned_deadline_day: Optional[int] = Field(None, ge=1, le=31)


class SubcategoryRecurringPatch(BaseModel):
    planned_amount: Optional[float] = None
    planned_deadline_day: Optional[int] = Field(None, ge=1, le=31)


class CalculateNetRequest(BaseModel):
    gross: float = Field(..., gt=0)
    contract_type: str = Field(..., min_length=1, max_length=64)


class CalculateNetResponse(BaseModel):
    net: float
    notes: str


class MonthlyTrendPoint(BaseModel):
    month: str
    income: float
    expenses: float
    net: float


class TrendsResponse(BaseModel):
    months: List[MonthlyTrendPoint]


class IncomeRowSync(BaseModel):
    """Full row snapshot for bulk sync (overwrites label + amounts for this id)."""

    id: int = Field(..., gt=0)
    label: str = Field(..., min_length=1, max_length=120)
    net_amount: Optional[Decimal] = None
    gross_amount: Optional[Decimal] = None


class RecurringRowSync(BaseModel):
    subcategory_id: int = Field(..., gt=0)
    planned_amount: Optional[float] = None
    planned_deadline_day: Optional[int] = Field(None, ge=1, le=31)  # None clears deadline in DB


class SettingsSyncRequest(BaseModel):
    """
    Persist all dashboard-driving inputs in one request (single DB transaction).

    Stored in: salary_rules, income_sources, subcategories (planned fields).
    """

    salary_day_of_month: int = Field(..., ge=1, le=31)
    target_ratio: float = Field(..., gt=0, lt=1)
    budget_strategy: str = Field(..., max_length=64)
    income_rows: List[IncomeRowSync] = Field(default_factory=list)
    recurring_rows: List[RecurringRowSync] = Field(default_factory=list)


class SettingsSyncResponse(BaseModel):
    salary_rule: SalaryRuleResponse
    income_rows_updated: int
    recurring_rows_updated: int
