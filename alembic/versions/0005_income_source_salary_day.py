from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_income_salary_day"
down_revision = "0004_income_budget"
branch_labels = None
depends_on = None


def _income_columns() -> set[str]:
    conn = op.get_bind()
    return {c["name"] for c in sa.inspect(conn).get_columns("income_sources")}


def upgrade() -> None:
    cols = _income_columns()
    if "salary_day_of_month" not in cols:
        op.add_column("income_sources", sa.Column("salary_day_of_month", sa.Integer(), nullable=True))


def downgrade() -> None:
    cols = _income_columns()
    if "salary_day_of_month" in cols:
        op.drop_column("income_sources", "salary_day_of_month")
