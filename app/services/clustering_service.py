"""
app/services/clustering_service.py
====================================
Phase 3C: BERTopic topic clustering for YouTube comments.

PIPELINE:
  1. Filter & dedup  — remove spam-only, short text, zero-like replies, dedup by text_hash
  2. Embed           — all-MiniLM-L6-v2 (singleton from relevance_filter)
  3. BERTopic fit    — UMAP(seed=42) + HDBSCAN(seed=42, dynamic min_cluster_size)
  4. Auto-adjust     — if outlier ratio > 35%, reduce min_cluster_size and refit (max 2x)
  5. Assign outliers — BERTopic reduce_outliers with embedding strategy
  6. Reduce topics   — cap at MAX_TARGET_CLUSTERS
  7. Post-process    — detect general_sentiment + bot_artifact cluster types
  8. Groq label      — single batch call for all cluster labels (4-5 words each)
  9. Gap analysis    — cosine sim of cluster label vs key_topics + key_claims

REPRODUCIBILITY: UMAP and HDBSCAN both use random_state=42.
Same input → same cluster IDs on every run.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from app.core.logging import get_logger
from app.services.relevance_filter import _get_model


def _to_int(v) -> int:
    """Convert numpy.int64 → Python int so MongoDB can serialize it."""
    return int(v)


def _to_float(v) -> float:
    """Convert numpy.float32/64 → Python float so MongoDB can serialize it."""
    return float(v)

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

INSUFFICIENT_DATA_THRESHOLD = 50
MAX_TARGET_CLUSTERS         = 20
MAX_OUTLIER_RATIO           = 0.35
MAX_ADJUST_ATTEMPTS         = 2
GAP_SIMILARITY_THRESHOLD    = 0.35
LABEL_CONFIDENCE_THRESHOLD  = 0.40
EMBED_BATCH_SIZE            = 128
GROQ_BASE_URL               = "https://api.groq.com/openai/v1"

# Top keywords that indicate a "generic sentiment" cluster — not a real topic
GENERIC_WORDS = frozenset({
    "great", "video", "good", "amazing", "thank", "thanks", "love",
    "nice", "best", "awesome", "wonderful", "helpful", "interesting",
    "excellent", "wow", "perfect", "fantastic", "incredible", "brilliant",
    "content", "channel", "like", "watch", "really", "very", "just",
})

INTENT_KEYS = [
    "question", "praise", "criticism", "confusion",
    "misconception", "request", "spam", "off_topic",
]


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ClusterDoc:
    cluster_id:          int
    label:               str
    keywords:            list[str]
    label_confidence:    float           # cosine(label_emb, mean keyword_emb)
    cluster_type:        str             # "topic" | "general_sentiment" | "bot_artifact"
    comment_count:       int
    is_content_gap:      bool
    gap_similarity_score: float          # max cosine sim to any key_topic/key_claim
    intent_breakdown:    dict
    sentiment_breakdown: dict
    top_comments:        list[dict]      # top 5 by like_count


@dataclass
class ClusteringResult:
    status:               str            # "completed" | "skipped_insufficient_data"
    clusters:             list[ClusterDoc] = field(default_factory=list)
    comment_assignments:  dict           = field(default_factory=dict)  # comment_id → cluster_id
    total_clustered:      int            = 0
    total_unclustered:    int            = 0
    outlier_ratio_before: float          = 0.0
    min_cluster_size_used: int           = 0
    error:                Optional[str]  = None


# ── Public API ────────────────────────────────────────────────────────────────

class ClusteringService:
    """
    Orchestrates the full Phase 3C clustering pipeline.

    Usage:
        svc = ClusteringService(api_key=settings.groq_api_key, model=settings.groq_model)
        result = await svc.cluster(classified_comments, summary)
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        self._api_key = api_key
        self._model   = model

    async def cluster(
        self,
        comments: list[dict],
        summary:  dict,
    ) -> ClusteringResult:
        # ── Step 1: Filter + dedup ────────────────────────────────────────
        to_cluster, excluded = _filter_and_dedup(comments)
        logger.info(
            "clustering_filter_done",
            total=len(comments), to_cluster=len(to_cluster), excluded=len(excluded),
        )

        if len(to_cluster) < INSUFFICIENT_DATA_THRESHOLD:
            logger.warning(
                "clustering_insufficient_data",
                count=len(to_cluster), threshold=INSUFFICIENT_DATA_THRESHOLD,
            )
            return ClusteringResult(
                status="skipped_insufficient_data",
                comment_assignments={c["comment_id"]: -1 for c in comments},
            )

        # ── Step 2: Embed ─────────────────────────────────────────────────
        embed_model = _get_model()
        texts       = [c["text"] for c in to_cluster]
        embeddings  = embed_model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        logger.info("clustering_embeddings_done", shape=list(embeddings.shape))

        # ── Steps 3–5: Fit + auto-adjust + assign outliers ───────────────
        topics, topic_model, min_size_used, outlier_ratio = _fit_with_auto_adjust(
            texts, embeddings
        )

        # ── Step 6: Reduce to MAX_TARGET_CLUSTERS ────────────────────────
        unique_topics = [t for t in set(topics) if t != -1]
        if len(unique_topics) > MAX_TARGET_CLUSTERS:
            logger.info(
                "clustering_reducing_topics",
                from_=len(unique_topics), to=MAX_TARGET_CLUSTERS,
            )
            topic_model.reduce_topics(texts, nr_topics=MAX_TARGET_CLUSTERS)
            topics = [_to_int(t) for t in topic_model.topics_]
            # Re-assign any outliers created by the merge
            topics = [_to_int(t) for t in topic_model.reduce_outliers(
                texts, topics, strategy="embeddings", embeddings=embeddings,
            )]
            # No update_topics — preserve clean fit_transform representations

        final_unique = [t for t in set(topics) if t != -1]
        logger.info("clustering_final_topics", count=len(final_unique))

        # ── Step 7: Build cluster docs + assignments ──────────────────────
        clusters, assignments = _build_clusters(
            to_cluster=to_cluster,
            topics=topics,
            topic_model=topic_model,
            excluded=excluded,
            all_comments=comments,
        )

        # ── Step 8: Detect artifact clusters ─────────────────────────────
        for cl in clusters:
            cl.cluster_type = _detect_cluster_type(cl, to_cluster, topics)

        # ── Step 9: Groq label ────────────────────────────────────────────
        clusters = await _label_clusters_groq(clusters, self._api_key, self._model, summary)

        # ── Step 10: Gap analysis ─────────────────────────────────────────
        _compute_gap_scores(clusters, summary, embed_model)

        return ClusteringResult(
            status="completed",
            clusters=clusters,
            comment_assignments=assignments,
            total_clustered=sum(1 for v in assignments.values() if v != -1),
            total_unclustered=sum(1 for v in assignments.values() if v == -1),
            outlier_ratio_before=outlier_ratio,
            min_cluster_size_used=min_size_used,
        )


# ── Step 1: Filter + dedup ───────────────────────────────────────────────────

def _filter_and_dedup(
    comments: list[dict],
) -> tuple[list[dict], dict[str, str]]:
    """
    Filter comments for clustering and deduplicate by text_hash.

    Returns:
        (to_cluster, excluded)
        excluded: {comment_id: reason} — for assignment fallback logic
    """
    seen_hashes: set[str] = set()
    to_cluster: list[dict] = []
    excluded: dict[str, str] = {}

    for c in comments:
        cid    = c["comment_id"]
        text   = (c.get("text") or "").strip()
        labels = set(c.get("intent_labels") or [])

        # Must be classified
        if c.get("classification_status") != "done":
            excluded[cid] = "not_classified"
            continue

        # Skip comments that are ONLY spam/off_topic — no topic signal
        if labels and labels <= {"spam", "off_topic"}:
            excluded[cid] = "spam_or_off_topic"
            continue

        # Minimum text length (replies can be shorter but still meaningful)
        min_len = 15 if c.get("is_reply") else 20
        if len(text) < min_len:
            excluded[cid] = "too_short"
            continue

        # Zero-like replies are conversational noise
        if c.get("is_reply") and (c.get("like_count") or 0) < 1:
            excluded[cid] = "zero_like_reply"
            continue

        # Deduplicate by text_hash (catches bot/spam copies)
        text_hash = c.get("text_hash") or hashlib.md5(text.encode()).hexdigest()
        if text_hash in seen_hashes:
            excluded[cid] = "duplicate"
            continue
        seen_hashes.add(text_hash)

        to_cluster.append(c)

    return to_cluster, excluded


# ── Steps 3–5: BERTopic fit + auto-adjust + outlier assignment ──────────────

def _build_topic_model(min_cluster_size: int) -> BERTopic:
    umap_model = UMAP(
        n_neighbors  = 15,
        n_components = 5,
        min_dist     = 0.0,
        metric       = "cosine",
        random_state = 42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size         = min_cluster_size,
        min_samples              = 5,
        metric                   = "euclidean",
        cluster_selection_method = "eom",
        prediction_data          = True,
    )
    # Remove stop words + allow bigrams so keywords are meaningful phrases
    # e.g. "morning sunlight" instead of "and", "the", "my"
    vectorizer_model = CountVectorizer(
        stop_words  = "english",
        min_df      = 2,           # word must appear in ≥2 docs to count
        ngram_range = (1, 2),      # unigrams + bigrams
    )
    return BERTopic(
        umap_model              = umap_model,
        hdbscan_model           = hdbscan_model,
        vectorizer_model        = vectorizer_model,
        calculate_probabilities = True,
        verbose                 = False,
        nr_topics               = "auto",
    )


def _fit_with_auto_adjust(
    texts:      list[str],
    embeddings: np.ndarray,
) -> tuple[list[int], BERTopic, int, float]:
    """
    Fit BERTopic, auto-adjusting min_cluster_size if too many outliers.

    Returns:
        (topics, topic_model, min_cluster_size_used, outlier_ratio_before_assignment)
    """
    n = len(texts)
    min_cluster_size = max(10, n // 80)
    outlier_ratio    = 1.0
    topic_model      = None
    topics           = []

    for attempt in range(MAX_ADJUST_ATTEMPTS + 1):
        topic_model = _build_topic_model(min_cluster_size)
        topics, _   = topic_model.fit_transform(texts, embeddings=embeddings)
        topics      = [_to_int(t) for t in topics]

        n_outliers    = topics.count(-1)
        outlier_ratio = n_outliers / n

        logger.info(
            "clustering_fit_attempt",
            attempt=attempt + 1,
            min_cluster_size=min_cluster_size,
            n_topics=len([t for t in set(topics) if t != -1]),
            outlier_ratio=round(outlier_ratio, 3),
        )

        if outlier_ratio <= MAX_OUTLIER_RATIO or attempt == MAX_ADJUST_ATTEMPTS:
            break

        # Reduce min_cluster_size by 30% and retry
        min_cluster_size = max(5, int(min_cluster_size * 0.7))

    # Assign outliers to nearest cluster by embedding distance.
    # NOTE: we intentionally do NOT call update_topics() here — it would
    # overwrite the clean CountVectorizer keyword representations from
    # fit_transform with mixed-bag c-TF-IDF from the now-enlarged clusters,
    # causing generic stop words to dominate large catch-all clusters.
    if -1 in topics:
        topics = [_to_int(t) for t in topic_model.reduce_outliers(
            texts, topics, strategy="embeddings", embeddings=embeddings,
        )]
        logger.info(
            "clustering_outliers_assigned",
            remaining_outliers=topics.count(-1),
        )

    return topics, topic_model, min_cluster_size, outlier_ratio


# ── Step 7: Build cluster docs + comment assignments ─────────────────────────

def _build_clusters(
    to_cluster:   list[dict],
    topics:       list[int],
    topic_model:  BERTopic,
    excluded:     dict[str, str],
    all_comments: list[dict],
) -> tuple[list[ClusterDoc], dict[str, int]]:
    """
    Build ClusterDoc objects and the full comment→cluster_id assignment map.
    """
    # ── Group comments by cluster_id ──────────────────────────────────────
    cluster_map: dict[int, list[dict]] = {}
    clustered_assignments: dict[str, int] = {}

    for comment, topic_id in zip(to_cluster, topics):
        cid = comment["comment_id"]
        native_id = _to_int(topic_id)
        clustered_assignments[cid] = native_id
        if native_id != -1:
            cluster_map.setdefault(native_id, []).append(comment)

    # ── Assign excluded comments ──────────────────────────────────────────
    # Zero-like replies → inherit parent's cluster_id
    # Everything else  → -1
    full_assignments: dict[str, int] = dict(clustered_assignments)
    for c in all_comments:
        cid = c["comment_id"]
        if cid in full_assignments:
            continue
        reason = excluded.get(cid, "not_classified")
        if reason == "zero_like_reply" and c.get("parent_comment_id"):
            parent_cluster = clustered_assignments.get(c["parent_comment_id"], -1)
            full_assignments[cid] = parent_cluster
        else:
            full_assignments[cid] = -1

    # ── Build ClusterDoc per cluster ──────────────────────────────────────
    clusters: list[ClusterDoc] = []
    for cluster_id, c_comments in sorted(cluster_map.items()):
        raw_keywords = topic_model.get_topic(cluster_id) or []
        keywords = [word for word, _ in raw_keywords[:10]]

        intent_bd, sentiment_bd = _compute_breakdowns(c_comments)
        top5 = sorted(c_comments, key=lambda x: x.get("like_count") or 0, reverse=True)[:5]

        clusters.append(ClusterDoc(
            cluster_id          = _to_int(cluster_id),
            label               = "",          # filled by Groq
            keywords            = keywords,
            label_confidence    = 0.0,         # filled after Groq
            cluster_type        = "topic",     # may be overwritten in Step 8
            comment_count       = len(c_comments),
            is_content_gap      = False,       # filled in Step 10
            gap_similarity_score = 0.0,        # filled in Step 10
            intent_breakdown    = intent_bd,
            sentiment_breakdown = sentiment_bd,
            top_comments        = [_slim_comment(c) for c in top5],
        ))

    return clusters, full_assignments


def _compute_breakdowns(comments: list[dict]) -> tuple[dict, dict]:
    n = len(comments) or 1
    intent_counts:    dict[str, int] = {k: 0 for k in INTENT_KEYS}
    sentiment_counts: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}

    for c in comments:
        for label in (c.get("intent_labels") or []):
            if label in intent_counts:
                intent_counts[label] += 1
        s = c.get("sentiment", "neutral")
        if s in sentiment_counts:
            sentiment_counts[s] += 1

    intent_bd    = {k: {"count": v, "pct": round(v / n * 100, 1)}
                    for k, v in intent_counts.items() if v > 0}
    sentiment_bd = {k: {"count": v, "pct": round(v / n * 100, 1)}
                    for k, v in sentiment_counts.items() if v > 0}
    return intent_bd, sentiment_bd


def _slim_comment(c: dict) -> dict:
    return {
        "comment_id":    c["comment_id"],
        "text":          (c.get("text") or "")[:300],
        "author_name":   c.get("author_name", ""),
        "like_count":    c.get("like_count") or 0,
        "intent_labels": c.get("intent_labels") or [],
        "sentiment":     c.get("sentiment", "neutral"),
    }


# ── Step 8: Detect artifact clusters ─────────────────────────────────────────

def _detect_cluster_type(
    cl:         ClusterDoc,
    to_cluster: list[dict],
    topics:     list[int],
) -> str:
    total = sum(1 for t in topics if t != -1)

    # Large cluster with generic keywords → general sentiment noise
    if cl.comment_count > total * 0.25:
        generic_overlap = len(set(kw.lower() for kw in cl.keywords[:5]) & GENERIC_WORDS)
        if generic_overlap >= 3:
            return "general_sentiment"

    # Cluster dominated by ≤3 unique authors → likely bot artifact
    comment_ids_in_cluster = {
        c["comment_id"]
        for c, t in zip(to_cluster, topics)
        if t == cl.cluster_id
    }
    if len(comment_ids_in_cluster) >= 5:
        authors = {
            c.get("author_channel_id") or c.get("author_name", "")
            for c in to_cluster
            if c["comment_id"] in comment_ids_in_cluster
        }
        unique_authors = sum(1 for a in authors if a)
        if unique_authors <= 3:
            return "bot_artifact"

    return "topic"


# ── Step 9: Groq label all clusters ──────────────────────────────────────────

async def _label_clusters_groq(
    clusters: list[ClusterDoc],
    api_key:  str,
    model:    str,
    summary:  dict,
) -> list[ClusterDoc]:
    if not clusters:
        return clusters

    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    # Build video context for grounding
    video_overview = summary.get("overview", "")
    key_topics     = [t.get("label", "") for t in summary.get("key_topics", []) if t.get("label")]
    topics_str     = ", ".join(key_topics) if key_topics else "not available"

    # Build cluster block
    cluster_descriptions = []
    for cl in clusters:
        samples = [c["text"][:120] for c in cl.top_comments[:3]]
        top_intent = max(cl.intent_breakdown.items(), key=lambda x: x[1]["count"])[0] \
                     if cl.intent_breakdown else "mixed"
        cluster_descriptions.append(
            f"Cluster {cl.cluster_id}:\n"
            f"  Top keywords: {', '.join(cl.keywords[:8])}\n"
            f"  Dominant intent: {top_intent}\n"
            f"  Sample comments:\n"
            + "\n".join(f"    - \"{s}\"" for s in samples)
        )

    system_prompt = (
        "You are an audience intelligence analyst for YouTube educational creators. "
        "Your job is to label comment topic clusters with SHORT, SPECIFIC, ACTIONABLE labels "
        "that tell the creator exactly what their audience is discussing.\n\n"
        "RULES:\n"
        "1. Labels must be 3-6 words — specific enough to be actionable\n"
        "2. Use the audience's own language and terminology\n"
        "3. Focus on WHAT they are asking/saying, not HOW they feel\n"
        "4. NEVER use generic labels like: 'General Discussion', 'Positive Feedback', "
        "'User Comments', 'Mixed Reactions', 'Various Topics'\n"
        "5. If comments are asking about something, start with the topic noun "
        "(e.g. 'Screen Time Questions' not 'Questions About Screens')\n\n"
        "FEW-SHOT EXAMPLES:\n"
        "---\n"
        "Keywords: screen, time, hours, computer, work, daily, staring\n"
        "Dominant intent: question\n"
        "Samples: \"I spend 8 hours on screens and my eyes are always dry\"\n"
        "         \"Does screen time actually damage vision permanently?\"\n"
        "→ Label: \"Screen Time Eye Damage\"\n\n"
        "Keywords: morning, sunlight, outside, natural, light, sun, minutes\n"
        "Dominant intent: praise\n"
        "Samples: \"Started going outside every morning for 20 min — vision feels clearer!\"\n"
        "         \"Natural sunlight first thing in the morning changed everything for me\"\n"
        "→ Label: \"Morning Sunlight Benefits\"\n\n"
        "Keywords: nac, drops, eye, carnosine, cataracts, using, tried\n"
        "Dominant intent: question\n"
        "Samples: \"Has anyone tried NAC eye drops for cataracts? Getting real results?\"\n"
        "         \"Where do you buy NAC drops and what brand is best?\"\n"
        "→ Label: \"NAC Eye Drop Results\"\n\n"
        "Keywords: supplement, lutein, zeaxanthin, dose, mg, take, vitamin\n"
        "Dominant intent: question\n"
        "Samples: \"What's the right lutein dosage? 10mg or 20mg?\"\n"
        "         \"Can I stack lutein with bilberry extract safely?\"\n"
        "→ Label: \"Lutein Supplement Dosage\"\n\n"
        "Keywords: diet, sugar, inflammation, food, eating, keto, insulin\n"
        "Dominant intent: praise\n"
        "Samples: \"Cutting sugar completely cleared my eye floaters in 3 weeks\"\n"
        "         \"Keto diet improved my dry eyes more than any eye drop\"\n"
        "→ Label: \"Diet & Eye Inflammation\"\n"
        "---"
    )

    user_prompt = (
        f"VIDEO OVERVIEW: {video_overview}\n"
        f"VIDEO KEY TOPICS: {topics_str}\n\n"
        "Label each of the following comment clusters using the rules above.\n\n"
        + "\n\n".join(cluster_descriptions)
        + '\n\nReturn ONLY this JSON (no explanation): '
        '{"labels": {"<cluster_id>": "<label>", ...}}'
    )

    try:
        response = await client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format = {"type": "json_object"},
            max_tokens  = 800,
            temperature = 0.2,   # low temp = consistent, specific labels
        )
        data   = json.loads(response.choices[0].message.content)
        labels = data.get("labels", {})
        logger.info("clustering_groq_labels_received", count=len(labels))
    except Exception as exc:
        logger.warning("clustering_groq_label_failed", error=str(exc))
        labels = {}

    # Assign labels + compute label_confidence
    embed_model = _get_model()
    for cl in clusters:
        raw_label = labels.get(str(cl.cluster_id), "").strip()
        if not raw_label:
            # Fallback: join top 3 clean keywords
            raw_label = " ".join(cl.keywords[:3]).title()

        cl.label = raw_label

        # label_confidence: cosine sim between label embedding and mean keyword embedding
        # With clean keywords (no stop words), this will now be meaningful
        if cl.keywords:
            try:
                label_emb    = embed_model.encode([cl.label], convert_to_numpy=True)
                keyword_embs = embed_model.encode(cl.keywords[:8], convert_to_numpy=True)
                mean_kw_emb  = keyword_embs.mean(axis=0, keepdims=True)
                sim          = _to_float(cosine_similarity(label_emb, mean_kw_emb)[0][0])
                cl.label_confidence = round(sim, 3)
            except Exception:
                cl.label_confidence = 0.0

    return clusters


# ── Step 10: Gap analysis ─────────────────────────────────────────────────────

def _compute_gap_scores(
    clusters:    list[ClusterDoc],
    summary:     dict,
    embed_model,
) -> None:
    """
    Compare each cluster label against the video's key_topics + key_claims.
    Low similarity → the video didn't cover this topic → content gap.
    Modifies clusters in-place.
    """
    topics = summary.get("key_topics", [])
    topic_texts = [
        f"{t.get('label', '')} {t.get('description', '')}".strip()
        for t in topics
        if t.get("label")
    ]
    claims        = summary.get("key_claims", [])
    reference     = topic_texts + claims

    if not reference:
        return

    ref_embeddings = embed_model.encode(
        reference,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    for cl in clusters:
        if not cl.label:
            continue
        label_emb = embed_model.encode(
            [cl.label],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        sims                    = cosine_similarity(label_emb, ref_embeddings)[0]
        max_sim                 = _to_float(sims.max())
        cl.gap_similarity_score = round(max_sim, 3)
        cl.is_content_gap       = max_sim < GAP_SIMILARITY_THRESHOLD
