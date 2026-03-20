from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from finance_tracker.api.routers import auth, dashboard, imports
from finance_tracker.config import get_cors_origin_list, get_session_secret_value
from finance_tracker.db.session import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Finance Tracker API", lifespan=lifespan)

    app.add_middleware(
        SessionMiddleware,
        secret_key=get_session_secret_value(),
        same_site="lax",
        https_only=False,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(dashboard.router)
    app.include_router(imports.router)

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
