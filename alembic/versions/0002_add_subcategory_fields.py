from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_subcategory_fields"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("subcategories", sa.Column("planned_amount", sa.Numeric(14, 2), nullable=True))
    op.add_column("subcategories", sa.Column("planned_deadline_day", sa.Integer(), nullable=True))
    op.add_column("subcategories", sa.Column("match_keywords", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("subcategories", "match_keywords")
    op.drop_column("subcategories", "planned_deadline_day")
    op.drop_column("subcategories", "planned_amount")

