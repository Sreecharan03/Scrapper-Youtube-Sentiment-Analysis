"""
app/api/v1/endpoints/videos.py
================================
FastAPI route handlers for video metadata.

ENDPOINTS:
  GET /api/v1/videos/{video_id} → Metadata for a scraped video
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.video import VideoResponse
from app.core.logging import get_logger
from app.db.connection import get_database
from app.db.repositories.video_repo import VideoRepository

router = APIRouter(prefix="/videos", tags=["Videos"])
logger = get_logger(__name__)


def get_video_repo(db: AsyncIOMotorDatabase = Depends(get_database)) -> VideoRepository:
    return VideoRepository(db)


@router.get(
    "/{video_id}",
    response_model=VideoResponse,
    summary="Get video metadata",
)
async def get_video(
    video_id: str,
    repo: VideoRepository = Depends(get_video_repo),
) -> VideoResponse:
    doc = await repo.get_by_video_id(video_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id!r} not found. Submit a scrape job first.",
        )
    return VideoResponse.from_document(doc)
