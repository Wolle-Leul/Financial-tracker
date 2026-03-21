from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query, Request

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.seed import ensure_demo_categories_seeded
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.dashboard import DashboardResponse
from finance_tracker.services.dashboard_service import build_dashboard_response

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(
    request: Request,
    year: int = Query(..., ge=2000, le=2100),
    month: int = Query(..., ge=1, le=12),
    categories: Optional[List[str]] = Query(None),
    subcategories: Optional[List[str]] = Query(None),
) -> DashboardResponse:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    ensure_demo_categories_seeded(user_id)
    return build_dashboard_response(
        user_id,
        year=year,
        month=month,
        selected_categories=categories,
        selected_subcategories=subcategories,
    )
