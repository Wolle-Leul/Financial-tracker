from __future__ import annotations

from sqlalchemy import select

from finance_tracker.db.models import User
from finance_tracker.db.session import session_scope


def get_or_create_default_user_id() -> int:
    """
    The app currently uses a single password gate, so we map that to one DB user row.
    """
    with session_scope() as session:
        user = session.execute(select(User).limit(1)).scalar_one_or_none()
        if user is None:
            user = User()
            session.add(user)
            session.flush()
        return int(user.id)

