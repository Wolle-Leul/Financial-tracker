from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Currently the app uses a password gate, so this is mostly future-proofing.


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    user: Mapped[User] = relationship("User")
    subcategories: Mapped[list["SubCategory"]] = relationship("SubCategory", back_populates="category")

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_categories_user_name"),)


class SubCategory(Base):
    __tablename__ = "subcategories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    # Planned/budget defaults used by the current dashboard until metrics are fully DB-driven.
    planned_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    planned_deadline_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Comma-separated keywords used for rules-based auto-mapping during import.
    match_keywords: Mapped[str | None] = mapped_column(Text, nullable=True)

    category: Mapped[Category] = relationship("Category", back_populates="subcategories")

    __table_args__ = (UniqueConstraint("category_id", "name", name="uq_subcategories_category_name"),)


class SalaryRule(Base):
    __tablename__ = "salary_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    salary_day_of_month: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    holiday_country: Mapped[str] = mapped_column(String(60), nullable=False, default="Poland")
    target_ratio: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, default=0.45)
    # Presets: custom_target_ratio, classic_50_30_20, zero_based, salary_window_only
    budget_strategy: Mapped[str] = mapped_column(String(64), nullable=False, default="custom_target_ratio")

    user: Mapped[User] = relationship("User")


class IncomeSource(Base):
    __tablename__ = "income_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(120), nullable=False, default="Income")
    employer_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_type: Mapped[str] = mapped_column(String(64), nullable=False, default="other")
    gross_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    net_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    use_net_only: Mapped[bool] = mapped_column(default=True, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    user: Mapped[User] = relationship("User")


class Import(Base):
    __tablename__ = "imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Import quality / parsing stats (used for KPIs and traceability).
    rows_parsed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parse_warnings_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    categorized_txn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    uncategorized_txn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    import_id: Mapped[int | None] = mapped_column(ForeignKey("imports.id", ondelete="SET NULL"), nullable=True)

    txn_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    merchant: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USD")

    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id", ondelete="SET NULL"), nullable=True)
    subcategory_id: Mapped[int | None] = mapped_column(ForeignKey("subcategories.id", ondelete="SET NULL"), nullable=True)

    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("user_id", "import_id", "txn_datetime", "amount", name="uq_txn_user_import_dt_amt"),
    )

