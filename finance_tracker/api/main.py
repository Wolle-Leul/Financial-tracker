from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from finance_tracker.api.routers import analytics, auth, dashboard, imports, incomes, recurring, settings
from finance_tracker.api.routers import tax_api
from finance_tracker.config import (
    get_cors_origin_list,
    get_session_same_site,
    get_session_secret_value,
)
from finance_tracker.db.ensure_schema import ensure_schema_extensions
from finance_tracker.db.migrate_up import migrate_upgrade
from finance_tracker.db.session import get_engine

_uvicorn_log = logging.getLogger("uvicorn.error")


def _session_middleware_kwargs() -> dict:
    """
    Hostinger (or any) SPA on domain A + API on Render domain B is cross-site.
    Browsers only send session cookies on those requests if SameSite=None and Secure.

    Set on Render: SESSION_SAME_SITE=none (and use HTTPS on the API).
    Local dev: leave unset (defaults to lax).
    """
    same_site = get_session_same_site()
    https_only = same_site == "none" or os.getenv("SESSION_HTTPS_ONLY", "").lower() == "true"
    return {"same_site": same_site, "https_only": https_only}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        migrate_upgrade()
    except Exception as e:
        _uvicorn_log.warning("Alembic upgrade failed (will try ensure_schema): %s", e)
    try:
        ensure_schema_extensions()
    except Exception as e:
        _uvicorn_log.exception("ensure_schema_extensions failed: %s", e)
        raise
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Finance Tracker API", lifespan=lifespan)

    session_kw = _session_middleware_kwargs()
    _origins = get_cors_origin_list()
    _uvicorn_log.info(
        "session cookie: same_site=%s https_only=%s | CORS allow_origins count=%s (cross-site: SESSION_SAME_SITE=none + CORS_ORIGINS=https://your-spa-host)",
        session_kw.get("same_site"),
        session_kw.get("https_only"),
        len(_origins),
    )
    app.add_middleware(
        SessionMiddleware,
        secret_key=get_session_secret_value(),
        **session_kw,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(dashboard.router)
    app.include_router(imports.router)
    app.include_router(settings.router)
    app.include_router(incomes.router)
    app.include_router(recurring.router)
    app.include_router(tax_api.router)
    app.include_router(analytics.router)

    @app.get("/")
    def root() -> dict[str, str]:
        """
        Browsers and uptime checks often GET /. This API has no HTML UI here;
        use the SPA (e.g. Hostinger) for the dashboard. Interactive API: /docs
        """
        return {
            "service": "Finance Tracker API",
            "docs": "/docs",
            "health": "/health",
            "auth_login": "POST /auth/login",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Database unavailable: {e}") from e
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"version": "0.1.0"}

    return app


app = create_app()
