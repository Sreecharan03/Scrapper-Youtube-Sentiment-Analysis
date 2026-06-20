"""
app/api/v1/schemas/clusters.py
================================
Pydantic response models for the clustering API.
"""

from typing import Optional
from pydantic import BaseModel


class ClusterTriggerResponse(BaseModel):
    video_id: str
    status:   str
    message:  str


class IntentCount(BaseModel):
    count: int
    pct:   float


class SentimentBreakdown(BaseModel):
    positive: Optional[IntentCount] = None
    neutral:  Optional[IntentCount] = None
    negative: Optional[IntentCount] = None


class TopComment(BaseModel):
    comment_id:    str
    text:          str
    author_name:   str
    like_count:    int
    intent_labels: list[str]
    sentiment:     str


class ClusterItem(BaseModel):
    cluster_id:          int
    label:               str
    keywords:            list[str]
    label_confidence:    float
    cluster_type:        str
    comment_count:       int
    is_content_gap:      bool
    gap_similarity_score: float
    intent_breakdown:    dict
    sentiment_breakdown: dict
    top_comments:        list[TopComment]

    @classmethod
    def from_document(cls, doc: dict) -> "ClusterItem":
        top = [
            TopComment(
                comment_id    = c.get("comment_id", ""),
                text          = c.get("text", ""),
                author_name   = c.get("author_name", ""),
                like_count    = c.get("like_count") or 0,
                intent_labels = c.get("intent_labels") or [],
                sentiment     = c.get("sentiment", "neutral"),
            )
            for c in (doc.get("top_comments") or [])
        ]
        return cls(
            cluster_id           = doc["cluster_id"],
            label                = doc.get("label", ""),
            keywords             = doc.get("keywords") or [],
            label_confidence     = doc.get("label_confidence") or 0.0,
            cluster_type         = doc.get("cluster_type", "topic"),
            comment_count        = doc.get("comment_count") or 0,
            is_content_gap       = doc.get("is_content_gap", False),
            gap_similarity_score = doc.get("gap_similarity_score") or 0.0,
            intent_breakdown     = doc.get("intent_breakdown") or {},
            sentiment_breakdown  = doc.get("sentiment_breakdown") or {},
            top_comments         = top,
        )


class ClusterInfoResponse(BaseModel):
    video_id:                    str
    status:                      str
    total_clusters:              int
    content_gap_count:           int
    comment_count_at_cluster_time: int
    total_clustered:             int
    total_unclustered:           int
    outlier_ratio_before:        float
    min_cluster_size_used:       int
    clustering_version:          str
    stale:                       bool
    computed_at:                 Optional[str] = None

    @classmethod
    def from_document(cls, doc: dict) -> "ClusterInfoResponse":
        completed_at = doc.get("completed_at")
        return cls(
            video_id                       = doc["video_id"],
            status                         = doc.get("status", "unknown"),
            total_clusters                 = doc.get("total_clusters") or 0,
            content_gap_count              = doc.get("content_gap_count") or 0,
            comment_count_at_cluster_time  = doc.get("comment_count_at_cluster_time") or 0,
            total_clustered                = doc.get("total_clustered") or 0,
            total_unclustered              = doc.get("total_unclustered") or 0,
            outlier_ratio_before           = doc.get("outlier_ratio_before") or 0.0,
            min_cluster_size_used          = doc.get("min_cluster_size_used") or 0,
            clustering_version             = doc.get("clustering_version", "v1"),
            stale                          = doc.get("stale", False),
            computed_at                    = completed_at.isoformat() if completed_at else None,
        )


class ClustersListResponse(BaseModel):
    video_id:         str
    total_clusters:   int
    content_gaps:     int
    stale:            bool
    clusters:         list[ClusterItem]
