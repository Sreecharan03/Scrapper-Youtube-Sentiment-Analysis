"""
app/api/v1/endpoints/recommendations.py
=========================================
Phase 3D — audience intelligence recommendation endpoints.

Routes:
  POST /recommendations/{video_id}        → trigger generation (async via Celery)
  GET  /recommendations/{video_id}/status → job status
  GET  /recommendations/{video_id}        → full result
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.recommendations import (
    RecommendationTriggerResponse,
    RecommendationsResponse,
)
from app.db.connection import get_database
from app.db.repositories.recommendation_repo import RecommendationRepository
from app.workers.tasks.recommendation_tasks import generate_recommendations

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def get_rec_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> RecommendationRepository:
    return RecommendationRepository(db)


@router.post(
    "/{video_id}",
    response_model=RecommendationTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_recommendations(
    video_id: str,
    rec_repo: RecommendationRepository = Depends(get_rec_repo),
):
    """Enqueue recommendation generation. Requires 3B + 3C to be complete."""
    current = await rec_repo.get_status(video_id)

    if current == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recommendation generation already in progress.",
        )

    generate_recommendations.delay(video_id)
    return RecommendationTriggerResponse(
        video_id=video_id,
        status="queued",
        message="Recommendation generation queued. Poll /status for progress.",
    )


@router.get("/{video_id}/status")
async def get_recommendation_status(
    video_id: str,
    rec_repo: RecommendationRepository = Depends(get_rec_repo),
):
    """Return current generation status."""
    doc = await rec_repo.get(video_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendation job found for this video.",
        )

    return {
        "video_id":     video_id,
        "status":       doc.get("status"),
        "generated_at": doc.get("generated_at"),
        "error":        doc.get("error"),
    }


@router.get("/{video_id}", response_model=RecommendationsResponse)
async def get_recommendations(
    video_id: str,
    rec_repo: RecommendationRepository = Depends(get_rec_repo),
):
    """Return full recommendation result."""
    doc = await rec_repo.get(video_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendations found. Trigger generation first.",
        )
    if doc.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Generation in progress or failed (status={doc.get('status')}).",
        )

    return RecommendationsResponse(
        video_id             = video_id,
        status               = doc["status"],
        generated_at         = doc.get("generated_at"),
        content_gaps         = doc.get("content_gaps", []),
        misconceptions       = doc.get("misconceptions", []),
        controversy_hotspots = doc.get("controversy_hotspots", []),
        unanswered_questions = doc.get("unanswered_questions", []),
        error                = doc.get("error"),
    )
