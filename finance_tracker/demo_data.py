from __future__ import annotations

import pandas as pd


def get_demo_expenses() -> pd.DataFrame:
    """
    Demo categories/subcategories used until PDF parsing + DB persistence land.
    """
    return pd.DataFrame(
        {
            "Category": ["Utilities", "Utilities", "Entertainment", "Housing", "Groceries"],
            "SubCategory": ["Phone", "Utilities", "You Tube", "Rent", "Groceries"],
            "Amount": [35.00, 70.00, 0.00, 2250.00, 800.00],
            # Day-of-month or "anchor" date; groceries deadline is adjusted later.
            "Deadline": [10, 12, 16, 12, 28],
        }
    )


def get_demo_incomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Source": ["Infy"],
            "FY": ["FY2324"],
            "Amount": [5100.00],
        }
    )

