"""
app/api/v1/endpoints/clusters.py
===================================
FastAPI endpoints for Phase 3C topic clustering.

ENDPOINTS:
  POST /api/v1/clusters/{video_id}          → dispatch cluster_comments task (202)
  GET  /api/v1/clusters/{video_id}          → list all clusters + stale flag
  GET  /api/v1/clusters/{video_id}/{id}     → single cluster detail
  GET  /api/v1/clusters/{video_id}/status   → job status only

PREREQUISITES: classification must be completed before clustering.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.clusters import (
    ClusterInfoResponse,
    ClusterItem,
    ClusterTriggerResponse,
    ClustersListResponse,
)
from app.core.logging import get_logger
from app.db.connection import get_database
from app.db.repositories.cluster_repo import ClusterRepository, ClusterStatus
from app.db.repositories.comment_repo import CommentRepository

router = APIRouter(prefix="/clusters", tags=["Clusters"])
logger = get_logger(__name__)


def get_cluster_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> ClusterRepository:
    return ClusterRepository(db)


def get_comment_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> CommentRepository:
    return CommentRepository(db)


@router.post(
    "/{video_id}",
    response_model = ClusterTriggerResponse,
    status_code    = status.HTTP_202_ACCEPTED,
    summary        = "Cluster comments for a video",
    description    = (
        "Dispatch BERTopic clustering on all classified comments. "
        "Returns 409 if clustering is already in progress. "
        "Returns 422 if classification is not yet completed."
    ),
)
async def trigger_clustering(
    video_id:     str,
    cluster_repo: ClusterRepository = Depends(get_cluster_repo),
    db:           AsyncIOMotorDatabase = Depends(get_database),
) -> ClusterTriggerResponse:

    # Guard: classification must be done
    analysis = await db["comment_analysis"].find_one(
        {"video_id": video_id, "status": "completed"}, {"_id": 0, "classified_count": 1}
    )
    if not analysis:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                f"Classification not completed for {video_id!r}. "
                "Run POST /api/v1/analysis/{video_id}/classify first."
            ),
        )

    # Guard: don't fire duplicate
    current_status = await cluster_repo.get_status(video_id)
    if current_status == ClusterStatus.PROCESSING:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = f"Clustering for {video_id!r} is already in progress.",
        )

    from app.workers.tasks.clustering_tasks import cluster_comments as _task
    _task.apply_async(kwargs={"video_id": video_id}, queue="scraper")

    logger.info("clustering_dispatched", video_id=video_id)

    return ClusterTriggerResponse(
        video_id = video_id,
        status   = "pending",
        message  = (
            f"Clustering queued for {video_id!r}. "
            f"Poll GET /api/v1/clusters/{video_id}/status for progress."
        ),
    )


@router.get(
    "/{video_id}/status",
    response_model = ClusterInfoResponse,
    summary        = "Get clustering job status",
)
async def get_cluster_status(
    video_id:     str,
    cluster_repo: ClusterRepository = Depends(get_cluster_repo),
    comment_repo: CommentRepository  = Depends(get_comment_repo),
) -> ClusterInfoResponse:

    doc = await cluster_repo.get_info(video_id)
    if not doc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                f"No clustering run found for {video_id!r}. "
                f"Submit POST /api/v1/clusters/{video_id} first."
            ),
        )

    # Auto-detect stale: if classified comment count grew > 10% since last run
    if doc.get("status") == ClusterStatus.COMPLETED and not doc.get("stale"):
        current_count = await comment_repo.count(
            {"video_id": video_id, "classification_status": "done"}
        )
        prev_count = doc.get("comment_count_at_cluster_time") or 0
        if prev_count > 0 and (current_count - prev_count) / prev_count > 0.10:
            await cluster_repo.mark_stale(video_id)
            doc["stale"] = True

    return ClusterInfoResponse.from_document(doc)


@router.get(
    "/{video_id}",
    response_model = ClustersListResponse,
    summary        = "List all topic clusters for a video",
    description    = (
        "Returns all clusters sorted by comment_count DESC. "
        "Includes stale flag if new comments have been classified since last clustering run."
    ),
)
async def list_clusters(
    video_id:     str,
    cluster_repo: ClusterRepository = Depends(get_cluster_repo),
    comment_repo: CommentRepository  = Depends(get_comment_repo),
) -> ClustersListResponse:

    info = await cluster_repo.get_info(video_id)
    if not info or info.get("status") not in (ClusterStatus.COMPLETED,):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                f"Clusters not available for {video_id!r}. "
                f"Submit POST /api/v1/clusters/{video_id} and wait for completion."
            ),
        )

    # Stale check
    stale = info.get("stale", False)
    if not stale:
        current_count = await comment_repo.count(
            {"video_id": video_id, "classification_status": "done"}
        )
        prev_count = info.get("comment_count_at_cluster_time") or 0
        if prev_count > 0 and (current_count - prev_count) / prev_count > 0.10:
            await cluster_repo.mark_stale(video_id)
            stale = True

    raw_clusters = await cluster_repo.get_clusters(video_id)
    clusters     = [ClusterItem.from_document(doc) for doc in raw_clusters]

    return ClustersListResponse(
        video_id       = video_id,
        total_clusters = len(clusters),
        content_gaps   = sum(1 for cl in clusters if cl.is_content_gap),
        stale          = stale,
        clusters       = clusters,
    )


@router.get(
    "/{video_id}/{cluster_id}",
    response_model = ClusterItem,
    summary        = "Get a single cluster by ID",
)
async def get_single_cluster(
    video_id:     str,
    cluster_id:   int,
    cluster_repo: ClusterRepository = Depends(get_cluster_repo),
) -> ClusterItem:

    doc = await cluster_repo.get_cluster(video_id, cluster_id)
    if not doc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"Cluster {cluster_id} not found for video {video_id!r}.",
        )
    return ClusterItem.from_document(doc)
