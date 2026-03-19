from __future__ import annotations

from alembic import op


def upgrade() -> None:
    # Initial migration: create all tables from SQLAlchemy models.
    from finance_tracker.db.models import Base

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    # Warning: this will drop all tables for the app.
    from finance_tracker.db.models import Base

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)

