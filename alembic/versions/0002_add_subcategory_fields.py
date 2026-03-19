from __future__ import annotations

import sqlalchemy as sa
from alembic import op


def upgrade() -> None:
    op.add_column("subcategories", sa.Column("planned_amount", sa.Numeric(14, 2), nullable=True))
    op.add_column("subcategories", sa.Column("planned_deadline_day", sa.Integer(), nullable=True))
    op.add_column("subcategories", sa.Column("match_keywords", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("subcategories", "match_keywords")
    op.drop_column("subcategories", "planned_deadline_day")
    op.drop_column("subcategories", "planned_amount")

