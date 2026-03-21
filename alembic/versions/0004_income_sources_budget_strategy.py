from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_income_budget"
down_revision = "0003_import_quality"
branch_labels = None
depends_on = None


def _salary_rule_columns() -> set[str]:
    conn = op.get_bind()
    return {c["name"] for c in sa.inspect(conn).get_columns("salary_rules")}


def upgrade() -> None:
    cols = _salary_rule_columns()
    if "budget_strategy" not in cols:
        op.add_column(
            "salary_rules",
            sa.Column("budget_strategy", sa.String(length=64), nullable=False, server_default="custom_target_ratio"),
        )

    conn = op.get_bind()
    if not sa.inspect(conn).has_table("income_sources"):
        op.create_table(
            "income_sources",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("label", sa.String(length=120), nullable=False, server_default="Income"),
            sa.Column("employer_name", sa.String(length=255), nullable=True),
            sa.Column("contract_type", sa.String(length=64), nullable=False, server_default="other"),
            sa.Column("gross_amount", sa.Numeric(14, 2), nullable=True),
            sa.Column("net_amount", sa.Numeric(14, 2), nullable=True),
            sa.Column("use_net_only", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_income_sources_user_id", "income_sources", ["user_id"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if insp.has_table("income_sources"):
        idx_names = {ix["name"] for ix in insp.get_indexes("income_sources")}
        if "ix_income_sources_user_id" in idx_names:
            op.drop_index("ix_income_sources_user_id", table_name="income_sources")
        op.drop_table("income_sources")
    cols = _salary_rule_columns()
    if "budget_strategy" in cols:
        op.drop_column("salary_rules", "budget_strategy")
