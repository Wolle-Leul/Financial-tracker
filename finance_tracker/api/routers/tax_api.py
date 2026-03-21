from __future__ import annotations

from fastapi import APIRouter, Request

from finance_tracker.api.deps import require_session_auth
from finance_tracker.schemas.settings import CalculateNetRequest, CalculateNetResponse
from finance_tracker.tax.pl_gross_net import estimate_net_from_gross

router = APIRouter(prefix="/api", tags=["tax"])


@router.post("/calculate-net", response_model=CalculateNetResponse)
def post_calculate_net(request: Request, body: CalculateNetRequest) -> CalculateNetResponse:
    require_session_auth(request)
    est = estimate_net_from_gross(body.gross, body.contract_type)
    return CalculateNetResponse(net=est.net, notes=est.notes)
