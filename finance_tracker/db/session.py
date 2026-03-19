from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from finance_tracker.config import get_config
from finance_tracker.db.models import Base


def get_engine():
    cfg = get_config()
    return create_engine(cfg.db_url, pool_pre_ping=True)


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db_for_dev() -> None:
    """
    Create tables directly (good for local/dev).

    For production, use Alembic migrations instead.
    """
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

