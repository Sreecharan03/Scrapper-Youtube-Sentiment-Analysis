"""
app/api/v1/endpoints/dashboard.py
===================================
Phase 3F — single aggregated dashboard endpoint.

GET /api/v1/dashboard/{video_id}
  Returns everything the Sighnal UI needs in one call.
  No LLM calls — pure aggregation from existing collections.
  Requires classification to be completed; other stages degrade gracefully.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.dashboard import DashboardResponse
from app.db.connection import get_database
from app.services.dashboard_service import build_dashboard

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/{video_id}", response_model=DashboardResponse)
async def get_dashboard(
    video_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Aggregate all analysis results for a video into one dashboard payload.

    Requires:
      - Classification completed (POST /api/v1/analysis/{video_id})

    Degrades gracefully if recommendations or intent summaries are not yet run
    (those sections return empty, pipeline_status reflects what's missing).
    """
    try:
        data = await build_dashboard(video_id, db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return DashboardResponse(**data)
