from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_subcategory_fields"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _subcategory_columns() -> set[str]:
    conn = op.get_bind()
    return {c["name"] for c in sa.inspect(conn).get_columns("subcategories")}


def upgrade() -> None:
    cols = _subcategory_columns()
    if "planned_amount" not in cols:
        op.add_column("subcategories", sa.Column("planned_amount", sa.Numeric(14, 2), nullable=True))
    if "planned_deadline_day" not in cols:
        op.add_column("subcategories", sa.Column("planned_deadline_day", sa.Integer(), nullable=True))
    if "match_keywords" not in cols:
        op.add_column("subcategories", sa.Column("match_keywords", sa.Text(), nullable=True))


def downgrade() -> None:
    cols = _subcategory_columns()
    if "match_keywords" in cols:
        op.drop_column("subcategories", "match_keywords")
    if "planned_deadline_day" in cols:
        op.drop_column("subcategories", "planned_deadline_day")
    if "planned_amount" in cols:
        op.drop_column("subcategories", "planned_amount")

