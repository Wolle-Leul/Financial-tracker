from __future__ import annotations

from fastapi import APIRouter, Query, Request

from finance_tracker.api.deps import require_session_auth
from finance_tracker.db.user import get_or_create_default_user_id
from finance_tracker.schemas.settings import TrendsResponse
from finance_tracker.services.analytics_service import build_monthly_trends

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics/trends", response_model=TrendsResponse)
def get_trends(request: Request, months: int = Query(12, ge=1, le=36)) -> TrendsResponse:
    require_session_auth(request)
    user_id = get_or_create_default_user_id()
    return build_monthly_trends(user_id, months=months)
