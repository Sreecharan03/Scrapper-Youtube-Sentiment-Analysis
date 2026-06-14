"""
app/services/relevance_filter.py
==================================
Embedding-based relevance filter for YouTube reply comments.

Before sending a reply to the LLM classifier, this filter checks whether
the reply is semantically related to:
  1. Its parent comment (is the reply actually responding to the parent?)
  2. The video's key topics + key claims (is the reply on-topic for this video?)

If neither similarity exceeds the threshold, the reply is skipped — it's
likely a personal conversation, a random reaction, or a spam tangent that
drifted away from the video's subject.

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dim embeddings, ~80MB, no GPU needed
  - ~14k sentences/sec on CPU with batching
  - Same model planned for Phase 3B (BERTopic clustering)

Threshold: 0.25  (cosine similarity scale)
  - >0.6  → clearly related
  - 0.3–0.6 → somewhat related (pass)
  - 0.1–0.25 → loosely related (borderline)
  - <0.25 → unrelated → skip
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger

logger = get_logger(__name__)

RELEVANCE_THRESHOLD = 0.25
EMBED_BATCH_SIZE    = 128
MODEL_NAME          = "all-MiniLM-L6-v2"

# ── Singleton model loader ────────────────────────────────────────────────────

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.info("relevance_filter_loading_model", model=MODEL_NAME)
        _MODEL = SentenceTransformer(MODEL_NAME)
        logger.info("relevance_filter_model_ready", model=MODEL_NAME)
    return _MODEL


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine_sim_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Vectorised cosine similarity between one query vector and a corpus matrix.
    Returns a 1-D array of similarities (one per corpus row).
    """
    query_norm  = query / (np.linalg.norm(query) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return corpus_norm @ query_norm


# ── RelevanceFilter ──────────────────────────────────────────────────────────

class RelevanceFilter:
    """
    Filters reply comments by semantic relevance before LLM classification.

    Usage:
        rf = RelevanceFilter(summary, threshold=0.25)
        relevant, skipped_ids = rf.filter_replies(replies)
    """

    def __init__(self, summary: dict, threshold: float = RELEVANCE_THRESHOLD) -> None:
        self._threshold = threshold
        self._model     = _get_model()

        # Build video context texts: key_topics + key_claims
        topics = summary.get("key_topics", [])
        topic_texts = [
            f"{t['label']}: {t.get('description', '')}"
            for t in topics
        ]
        claims = summary.get("key_claims", [])
        video_context = topic_texts + claims

        if video_context:
            self._video_embeddings: np.ndarray = self._model.encode(
                video_context,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            self._video_embeddings = None

        logger.info(
            "relevance_filter_initialized",
            video_contexts=len(video_context),
            threshold=threshold,
        )

    def filter_replies(
        self,
        replies: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """
        Filter a list of reply comment dicts by semantic relevance.

        Args:
            replies: list of {comment_id, text, parent_text?, ...}
                     Only comments with is_reply=True should be passed here.

        Returns:
            (relevant_replies, skipped_comment_ids)
        """
        if not replies:
            return [], []

        # ── Batch encode all reply texts ──────────────────────────────────
        reply_texts = [r.get("text", "") for r in replies]
        reply_embeddings: np.ndarray = self._model.encode(
            reply_texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # ── Batch encode unique parent texts ──────────────────────────────
        unique_parents = list({r["parent_text"] for r in replies if r.get("parent_text")})
        parent_emb_map: dict[str, np.ndarray] = {}
        if unique_parents:
            parent_embeddings = self._model.encode(
                unique_parents,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            parent_emb_map = dict(zip(unique_parents, parent_embeddings))

        # ── Pre-compute each parent's own video-relevance score ──────────
        # Used to prevent off-topic reply threads from passing through
        # just because they're internally consistent (e.g. dog convo).
        parent_video_sim_map: dict[str, float] = {}
        if parent_emb_map and self._video_embeddings is not None:
            for pt, pe in parent_emb_map.items():
                sims = _cosine_sim_matrix(pe, self._video_embeddings)
                parent_video_sim_map[pt] = float(sims.max())

        # ── Score each reply ──────────────────────────────────────────────
        relevant:    list[dict] = []
        skipped_ids: list[str]  = []

        for reply, reply_emb in zip(replies, reply_embeddings):
            parent_text = reply.get("parent_text", "")
            parent_video_sim = parent_video_sim_map.get(parent_text, 0.0)
            if self._is_relevant(reply_emb, parent_text, parent_emb_map, parent_video_sim):
                relevant.append(reply)
            else:
                skipped_ids.append(reply["comment_id"])

        logger.info(
            "relevance_filter_done",
            total_replies=len(replies),
            relevant=len(relevant),
            skipped=len(skipped_ids),
            threshold=self._threshold,
        )
        return relevant, skipped_ids

    def _is_relevant(
        self,
        reply_emb: np.ndarray,
        parent_text: str,
        parent_emb_map: dict[str, np.ndarray],
        parent_video_sim: float = 0.0,
    ) -> bool:
        # ── Check 1: reply directly related to video topics/claims ────────
        video_sim = 0.0
        if self._video_embeddings is not None:
            sims = _cosine_sim_matrix(reply_emb, self._video_embeddings)
            video_sim = float(sims.max())

        if video_sim >= self._threshold:
            return True

        # ── Check 2: reply related to parent AND parent is video-relevant ─
        # This handles context-dependent replies like "exactly!" or "same"
        # that have low direct video similarity but respond to an on-topic parent.
        # IMPORTANT: we also verify the parent itself is video-relevant so that
        # off-topic reply threads (e.g. a dog conversation) don't pass through.
        if parent_text and parent_text in parent_emb_map:
            parent_emb = parent_emb_map[parent_text]
            parent_sim = float(_cosine_sim_matrix(reply_emb, parent_emb[np.newaxis, :])[0])
            if parent_sim >= 0.45 and parent_video_sim >= self._threshold:
                return True

        return False
