from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from sqlalchemy import select

from finance_tracker.db.models import Category, SubCategory
from finance_tracker.demo_data import get_demo_expenses


_DEFAULT_KEYWORDS: Dict[str, str] = {
    # Groceries
    "Groceries": "grocery,grocer,supermarket,market",
    # Housing
    "Rent": "rent,landlord,lease",
    # Entertainment
    "You Tube": "youtube,yt,tube",
    # Utilities
    "Phone": "phone,mobile,cell,operator",
    "Utilities": "utility,utilities,internet,energy,electric,water",
}


def ensure_demo_categories_seeded(user_id: int) -> None:
    """
    Seed DB with demo categories/subcategories so the dashboard can run before
    fully DB-driven budgets/transactions.
    """
    # Avoid creating duplicates.
    from finance_tracker.db.session import session_scope

    with session_scope() as session:
        existing = session.execute(select(Category).where(Category.user_id == user_id).limit(1)).scalar_one_or_none()
        if existing is not None:
            return

        demo_expenses = get_demo_expenses()
        # Create categories
        categories_by_name: Dict[str, Category] = {}
        for cat_name in sorted(demo_expenses["Category"].dropna().unique().tolist()):
            c = Category(user_id=user_id, name=str(cat_name))
            session.add(c)
            categories_by_name[str(cat_name)] = c
        session.flush()

        # Create subcategories
        for _, row in demo_expenses.iterrows():
            cat_name = str(row["Category"])
            sub_name = str(row["SubCategory"])
            planned_amount = float(row["Amount"]) if row.get("Amount") is not None else None
            planned_deadline_day = int(row["Deadline"]) if row.get("Deadline") is not None else None
            keywords = _DEFAULT_KEYWORDS.get(sub_name)

            session.add(
                SubCategory(
                    category_id=categories_by_name[cat_name].id,
                    name=sub_name,
                    planned_amount=planned_amount,
                    planned_deadline_day=planned_deadline_day,
                    match_keywords=keywords,
                )
            )

