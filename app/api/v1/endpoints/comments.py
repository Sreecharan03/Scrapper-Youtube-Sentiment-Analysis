"""
app/api/v1/endpoints/comments.py
==================================
FastAPI route handlers for comment retrieval.

ENDPOINTS:
  GET /api/v1/comments?video_id=...  → Paginated comments for a video
  GET /api/v1/comments/top?video_id= → Top comments by likes
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.comment import CommentListResponse, CommentResponse
from app.core.logging import get_logger
from app.db.connection import get_database
from app.db.repositories.comment_repo import CommentRepository

router = APIRouter(prefix="/comments", tags=["Comments"])
logger = get_logger(__name__)


def get_comment_repo(db: AsyncIOMotorDatabase = Depends(get_database)) -> CommentRepository:
    return CommentRepository(db)


@router.get(
    "",
    response_model=CommentListResponse,
    summary="Get comments for a video",
)
async def get_comments(
    video_id: str = Query(..., description="YouTube video ID (11 characters)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    sort_by: str = Query("published_at", description="Sort field"),
    descending: bool = Query(True, description="Newest/highest first"),
    repo: CommentRepository = Depends(get_comment_repo),
) -> CommentListResponse:
    comments = await repo.get_comments_for_video(
        video_id,
        skip=skip,
        limit=limit,
        sort_by=sort_by,
        descending=descending,
    )
    total = await repo.count_comments_for_video(video_id)

    return CommentListResponse(
        comments=[CommentResponse.from_document(c) for c in comments],
        video_id=video_id,
        total_stored=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/top",
    response_model=list[CommentResponse],
    summary="Get top comments by like count",
)
async def get_top_comments(
    video_id: str = Query(...),
    limit: int = Query(10, ge=1, le=100),
    repo: CommentRepository = Depends(get_comment_repo),
) -> list[CommentResponse]:
    comments = await repo.get_top_comments(video_id, limit=limit)
    return [CommentResponse.from_document(c) for c in comments]
