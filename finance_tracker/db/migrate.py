from __future__ import annotations

import subprocess
import sys
from typing import Optional


def upgrade_head(db_url: Optional[str] = None) -> None:
    """
    Apply Alembic migrations up to `head`.

    Intended for deployment/ops scripts, not for Streamlit request flow.
    """
    cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
    env = None
    if db_url:
        import os

        env = dict(**os.environ, DB_URL=db_url)
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    upgrade_head()

