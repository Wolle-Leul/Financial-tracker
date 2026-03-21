from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_import_quality"
down_revision = "0002_subcategory_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("imports", sa.Column("rows_parsed", sa.Integer(), nullable=False, server_default="0"))
    op.add_column(
        "imports",
        sa.Column("parse_warnings_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "imports",
        sa.Column("categorized_txn_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "imports",
        sa.Column("uncategorized_txn_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("imports", "uncategorized_txn_count")
    op.drop_column("imports", "categorized_txn_count")
    op.drop_column("imports", "parse_warnings_count")
    op.drop_column("imports", "rows_parsed")

