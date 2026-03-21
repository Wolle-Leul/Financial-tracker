from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from finance_tracker.config import get_config
from finance_tracker.db.models import Base


# Alembic Config object provides access to the values within alembic.ini.
config = context.config

# Ensure Alembic uses our configured DB URL.
# ConfigParser treats % as interpolation; escape for URL-encoded password segments (%2F, etc.).
_db_url = get_config().db_url or ""
config.set_main_option("sqlalchemy.url", _db_url.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

