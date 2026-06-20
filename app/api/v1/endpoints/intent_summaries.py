"""
app/api/v1/endpoints/intent_summaries.py

Routes:
  POST /intent-summaries/{video_id}        → trigger generation
  GET  /intent-summaries/{video_id}/status → job status
  GET  /intent-summaries/{video_id}        → full result (cached)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.intent_summaries import (
    IntentSummariesResponse,
    IntentSummaryTriggerResponse,
)
from app.db.connection import get_database
from app.db.repositories.intent_summary_repo import IntentSummaryRepository
from app.workers.tasks.intent_summary_tasks import generate_intent_summaries

router = APIRouter(prefix="/intent-summaries", tags=["intent-summaries"])


def get_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> IntentSummaryRepository:
    return IntentSummaryRepository(db)


@router.post(
    "/{video_id}",
    response_model=IntentSummaryTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger(
    video_id: str,
    repo: IntentSummaryRepository = Depends(get_repo),
):
    current = await repo.get_status(video_id)
    if current == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Intent summary generation already in progress.",
        )
    generate_intent_summaries.delay(video_id)
    return IntentSummaryTriggerResponse(
        video_id=video_id,
        status="queued",
        message="Intent summary generation queued. Poll /status for progress.",
    )


@router.get("/{video_id}/status")
async def get_status(
    video_id: str,
    repo: IntentSummaryRepository = Depends(get_repo),
):
    doc = await repo.get(video_id)
    if not doc:
        raise HTTPException(status_code=404, detail="No intent summary job found.")
    return {
        "video_id":     video_id,
        "status":       doc.get("status"),
        "generated_at": doc.get("generated_at"),
        "error":        doc.get("error"),
    }


@router.get("/{video_id}", response_model=IntentSummariesResponse)
async def get_summaries(
    video_id: str,
    repo: IntentSummaryRepository = Depends(get_repo),
):
    doc = await repo.get(video_id)
    if not doc:
        raise HTTPException(status_code=404, detail="No intent summaries found. Trigger generation first.")
    if doc.get("status") != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Generation in progress or failed (status={doc.get('status')}).",
        )
    return IntentSummariesResponse(
        video_id         = video_id,
        status           = doc["status"],
        generated_at     = doc.get("generated_at"),
        overall_summary  = doc.get("overall_summary", ""),
        intent_summaries = doc.get("intent_summaries", {}),
        error            = doc.get("error"),
    )
