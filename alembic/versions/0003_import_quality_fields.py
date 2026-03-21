from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_import_quality"
down_revision = "0002_subcategory_fields"
branch_labels = None
depends_on = None


def _import_columns() -> set[str]:
    conn = op.get_bind()
    return {c["name"] for c in sa.inspect(conn).get_columns("imports")}


def upgrade() -> None:
    cols = _import_columns()
    if "rows_parsed" not in cols:
        op.add_column("imports", sa.Column("rows_parsed", sa.Integer(), nullable=False, server_default="0"))
    if "parse_warnings_count" not in cols:
        op.add_column(
            "imports",
            sa.Column("parse_warnings_count", sa.Integer(), nullable=False, server_default="0"),
        )
    if "categorized_txn_count" not in cols:
        op.add_column(
            "imports",
            sa.Column("categorized_txn_count", sa.Integer(), nullable=False, server_default="0"),
        )
    if "uncategorized_txn_count" not in cols:
        op.add_column(
            "imports",
            sa.Column("uncategorized_txn_count", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    cols = _import_columns()
    if "uncategorized_txn_count" in cols:
        op.drop_column("imports", "uncategorized_txn_count")
    if "categorized_txn_count" in cols:
        op.drop_column("imports", "categorized_txn_count")
    if "parse_warnings_count" in cols:
        op.drop_column("imports", "parse_warnings_count")
    if "rows_parsed" in cols:
        op.drop_column("imports", "rows_parsed")

