# Sighnal — AI Content Strategist for YouTube Creators

Scrapes every comment (top-level + replies) from any YouTube video using YouTube's internal Innertube API — no API key, no quota limits — then runs a full NLP + LLM analysis pipeline to surface audience intent, topic clusters, content gaps, misconceptions, and actionable video recommendations. Results served via a single aggregated dashboard API.

---

## What It Does

A creator pastes a YouTube URL. The platform:

1. Scrapes all comments (handles 10K+ threads, resumes after crashes)
2. Fetches the transcript and generates a structured video summary
3. Classifies every comment by intent (question / praise / criticism / misconception / etc.) and sentiment
4. Clusters comments into topic groups using BERTopic
5. Detects content gaps, misconceptions, controversy hotspots, and unanswered questions
6. Generates ranked video ideas backed by real comment demand
7. Serves everything through one dashboard endpoint — health score, opportunity score, risk score, top 3 video ideas, intent tabs, and deep-dive analysis

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Task queue | Celery (`scraper` + `replies` + `recommendations` queues) |
| Database | MongoDB (local, persistent path) |
| Cache / broker | Redis |
| Scraping | aiohttp + YouTube Innertube API (no official API key) |
| Transcript | `youtube-transcript-api` v1.x |
| LLM summary + intent summaries | Anthropic Claude Haiku (`claude-haiku-4-5-20251001`) |
| Strategic recommendations | Groq `llama-3.3-70b-versatile` |
| Classification + cluster labeling + enrichment | Groq `llama-3.1-8b-instant` |
| Topic clustering | `bertopic==0.17.4` + UMAP + HDBSCAN |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim) |

---

## Project Status

```
Phase 1  — Architecture & infrastructure        ✅ Complete
Phase 2  — Comment scraping pipeline            ✅ Complete
Phase 3A — Transcript + LLM summary             ✅ Complete
Phase 3B — Comment classification               ✅ Complete
Phase 3C — BERTopic topic clustering            ✅ Complete
Phase 3D — Gap analysis + recommendations       ✅ Complete (production-grade prompts)
Phase 3E — Per-intent audience summaries        ✅ Complete
Phase 3F — Dashboard aggregation API            ✅ Complete
Phase 3G — React / Next.js Sighnal UI           🔜 Next
Phase 4  — Channel-level cross-video analysis   🔜 Planned
```

### Results on test video (eye health — 3,569 comments)

| Metric | Value |
|---|---|
| Comments scraped | 3,569 |
| Classified | 3,082 (86.3%) |
| Topic clusters | 13 |
| Content gaps | 1 (Astigmatism — 76% questions) |
| Controversy hotspots | 1 (LED Safety — 73% criticism, 805-like top comment) |
| Misconception clusters | 5 |
| Audience health score | 69 / 100 |
| Opportunity score | 95 / 100 |
| Risk score | 11 / 100 |
| Top praise | 41.2% of classified comments |

---

## Project Structure

```
app/
├── api/
│   ├── router.py
│   └── v1/
│       ├── endpoints/
│       │   ├── jobs.py                  # scrape job submit/status
│       │   ├── comments.py              # comment listing
│       │   ├── videos.py                # video metadata
│       │   ├── transcripts.py           # transcript fetch/get
│       │   ├── summaries.py             # LLM summary generate/get
│       │   ├── analysis.py              # classification + retry + results
│       │   ├── clusters.py              # BERTopic clustering (3C)
│       │   ├── recommendations.py       # gap analysis + recommendations (3D)
│       │   ├── intent_summaries.py      # per-intent audience summaries (3E)
│       │   └── dashboard.py             # aggregated dashboard endpoint (3F)
│       └── schemas/
│           ├── recommendations.py
│           └── dashboard.py             # DashboardResponse with all sections
├── core/
│   ├── config.py                        # groq_model + groq_strategic_model
│   └── logging.py
├── db/
│   ├── init_db.py
│   └── repositories/
│       ├── comment_repo.py
│       ├── classification_repo.py
│       ├── cluster_repo.py
│       ├── recommendation_repo.py
│       └── intent_summary_repo.py
├── services/
│   ├── classifier.py                    # Groq batched comment classifier (3B)
│   ├── clustering_service.py            # BERTopic pipeline (3C)
│   ├── recommendation_service.py        # two-model Groq strategy engine (3D)
│   ├── dashboard_service.py             # score computation + aggregation (3F)
│   ├── intent_summary_service.py        # per-intent LLM summaries (3E)
│   ├── relevance_filter.py              # all-MiniLM-L6-v2 reply filter
│   ├── text_preprocessor.py
│   └── emoji_dict.json                  # 1,918 emoji → Unicode map
└── workers/
    ├── celery_app.py
    └── tasks/
        ├── job_tasks.py
        ├── tlc_tasks.py
        ├── reply_tasks.py
        ├── transcript_tasks.py
        ├── summary_tasks.py
        ├── classification_tasks.py
        ├── clustering_tasks.py
        ├── recommendation_tasks.py
        └── intent_summary_tasks.py
```

---

## Setup

### Prerequisites

- Python 3.12
- MongoDB (local)
- Redis

### Install

```bash
pip install -r requirements.txt
```

### Environment variables

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=yt_scraper

# Redis
REDIS_HOST=your-redis-host
REDIS_PORT=6379

# Anthropic — summary (3A) + intent summaries (3E)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# Groq — classification (3B), cluster labeling (3C), enrichment (3D)
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant

# Groq — strategic recommendations layer (3D) — stronger reasoning
GROQ_STRATEGIC_MODEL=llama-3.3-70b-versatile
```

---

## Running

### 1. Start MongoDB (persistent path — survives restarts)

```bash
mongod --dbpath /teamspace/studios/this_studio/.mongodb/data \
       --logpath /teamspace/studios/this_studio/.mongodb/mongod.log \
       --fork
```

### 2. Start Celery worker (all 3 queues required)

```bash
celery -A app.workers.celery_app worker \
       -Q scraper,replies,recommendations \
       --concurrency=2 \
       --loglevel=INFO
```

### 3. Start FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

> ⚠️ Redis free tier caps at 30 connections. Kill the Celery worker before restarting uvicorn if Redis refuses connection.

---

## API Endpoints

### Scraping

```
POST /api/v1/jobs                                Submit scrape job for a video URL
GET  /api/v1/jobs/{job_id}                       Job status + progress
GET  /api/v1/comments?video_id=...               List scraped comments
```

### Phase 3A — Transcript + Summary

```
POST /api/v1/transcripts/{video_id}              Fetch transcript (manual → auto → any language)
GET  /api/v1/transcripts/{video_id}              Get stored transcript
POST /api/v1/summaries/{video_id}                Generate LLM video summary (draft → self-critique)
GET  /api/v1/summaries/{video_id}                Get stored summary
```

### Phase 3B — Classification

```
POST /api/v1/analysis/{video_id}/classify        Classify all comments (auto-retries 3x)
POST /api/v1/analysis/{video_id}/classify/retry  Retry only failed comments
GET  /api/v1/analysis/{video_id}                 Intent + sentiment breakdown
```

### Phase 3C — Topic Clustering

```
POST /api/v1/clusters/{video_id}                 Run BERTopic clustering
GET  /api/v1/clusters/{video_id}/status          Clustering job status
GET  /api/v1/clusters/{video_id}                 All clusters sorted by size
GET  /api/v1/clusters/{video_id}/{cluster_id}    Single cluster detail
```

### Phase 3D — Recommendations

```
POST /api/v1/recommendations/{video_id}          Generate recommendations (two-model Groq pipeline)
GET  /api/v1/recommendations/{video_id}/status   Job status
GET  /api/v1/recommendations/{video_id}          Full result — video ideas, gaps, misconceptions, controversies
```

### Phase 3E — Intent Summaries

```
POST /api/v1/intent-summaries/{video_id}         Generate per-intent summaries (cached)
GET  /api/v1/intent-summaries/{video_id}/status  Job status
GET  /api/v1/intent-summaries/{video_id}         All 8 intent summaries + overall summary
```

### Phase 3F — Dashboard

```
GET  /api/v1/dashboard/{video_id}                Full aggregated dashboard — one call, everything
```

---

## Full Pipeline (new video)

```bash
VIDEO_ID="nCnZX8zs4LI"
BASE="http://localhost:8000"

# 1. Scrape all comments
curl -X POST $BASE/api/v1/jobs \
     -H "Content-Type: application/json" \
     -d '{"video_url": "https://www.youtube.com/watch?v='$VIDEO_ID'"}'

# 2. Transcript + summary (after scrape completes)
curl -X POST $BASE/api/v1/transcripts/$VIDEO_ID
curl -X POST $BASE/api/v1/summaries/$VIDEO_ID

# 3. Classify (after summary completes)
curl -X POST $BASE/api/v1/analysis/$VIDEO_ID/classify

# 4. Cluster (after classification completes)
curl -X POST $BASE/api/v1/clusters/$VIDEO_ID

# 5. Recommendations + intent summaries (after clustering completes)
curl -X POST $BASE/api/v1/recommendations/$VIDEO_ID
curl -X POST $BASE/api/v1/intent-summaries/$VIDEO_ID

# 6. Dashboard — everything in one call
curl $BASE/api/v1/dashboard/$VIDEO_ID | python3 -m json.tool
```

---

## Analysis Pipeline

### Phase 3D — Recommendation Engine (Two-Model Architecture)

```
Call 1 — Strategic layer (llama-3.3-70b-versatile)
  · Quoted data block: all numbers injected verbatim — model cannot hallucinate stats
  · Server-side slot enumeration: _select_video_slots() pre-selects top 5 clusters
    by demand signal (gap clusters first, then highest question_count)
    LLM converts each slot to a video idea — it does not choose which clusters
  · CoT scratchpad: model fills "reasoning" field before committing to answers
  · Verification field: model echoes back locked stats; server validates against actual data
  · Output: executive_summary, audience_stage, audience_mood, top_video_ideas (5),
    purchase_intent_signals, content_series, risk_alerts

Call 2 — Per-item enrichment (llama-3.1-8b-instant)
  · Generates what_to_do / why / suggested_hook / urgency / impact_type per finding
  · Misconception hooks anchored to KEY CLAIMS only — no external science added
  · Controversy hooks never cite specific journal names or publication years
  · Urgency calibrated to comment count: ≥100=high, 20-99=medium, <20=low
```

### Phase 3F — Dashboard Scores

Three scores computed server-side from classification data — no LLM calls:

```
health_score      = positive_pct × 0.50
                  + (100 - misconception_pct) × 0.30
                  + (100 - criticism_pct) × 0.20

opportunity_score = top video idea's demand_score (0–100)
                    calibrated by 70b model against slot demand signals

risk_score        = criticism_pct × 0.60
                  + misconception_pct × 0.40
```

### Phase 3C — BERTopic Clustering

```
Filter + dedup (spam, len<20, text_hash dedup)
  ↓
Embed (all-MiniLM-L6-v2)
  ↓
BERTopic:
  UMAP(n_neighbors=15, n_components=5, min_dist=0, metric=cosine, seed=42)
  HDBSCAN(min_cluster_size=max(10, n//80), min_samples=5, prediction_data=True)
  CountVectorizer(stop_words=english, min_df=2, ngram_range=(1,2))
  ↓
Auto-adjust: outlier_ratio > 35% → min_cluster_size × 0.7, refit (max 2 rounds)
  ↓
reduce_outliers(strategy=embeddings)   ← never call update_topics() after this
  ↓
Groq few-shot labeling (5 domain examples + video context, temp=0.2)
  ↓
Gap detection: cosine_sim(cluster_label, key_topics + key_claims) < 0.35 → content gap
```

---

## MongoDB Collections

| Collection | Purpose |
|---|---|
| `comments` | All TLCs and replies with intent_labels, sentiment, cluster_id |
| `jobs` | Scrape job lifecycle + progress |
| `scrape_batches` | Per-batch checkpoint state |
| `scrape_sessions` | Resume tokens for interrupted scrapes |
| `comment_history` | Edit detection archive |
| `failed_replies` | Reply batches that exhausted all retries |
| `transcripts` | Video transcripts (all languages) |
| `summaries` | LLM-generated video summaries with key_claims + controversy_triggers |
| `comment_analysis` | Aggregate intent + sentiment breakdown per video |
| `reports` | Clean client-facing export per video |
| `clusters` | One doc per topic cluster with top_comments + intent_breakdown |
| `cluster_info` | Clustering job status + metadata |
| `recommendations` | Video ideas, gaps, misconceptions, controversies, strategic layer |
| `intent_summaries` | Per-intent LLM summaries, cached (re-triggers on >10% comment growth) |

---

## Key Engineering Notes

- **YouTube scraping**: Innertube `/youtubei/v1/next` with ViewModel+mutations format (2024+). Old `commentRenderer` format is legacy. Install `Brotli` — YouTube returns brotli-encoded responses.
- **BERTopic keyword quality**: Never call `update_topics()` after `reduce_outliers()`. It forces c-TF-IDF recompute on enlarged clusters, causing stop words to dominate keywords.
- **numpy → MongoDB**: BERTopic returns `numpy.int64`/`float64`. PyMongo rejects them — wrap every BERTopic output with `int()`/`float()` before storing.
- **Fan cluster filtering**: BERTopic types fan praise clusters as "topic". Filter clusters with praise > 50% from all recommendation analysis — do not re-cluster.
- **Groq 8b + JSON**: Always use `response_format={"type": "json_object"}`. Model wraps output in markdown fences without it.
- **Two-model Groq**: Run strategic call (70b) first, enrichment call (8b) second — sequentially. Concurrent calls hit rate limits and one fails silently.
- **mark_completed**: Always use `**data` spread in `update_one($set)`. A hardcoded field list silently drops any new fields added to the result.
- **Managed Redis**: Only DB 0 on Redis Cloud free tier; max 30 connections. Use `BlockingConnectionPool(max_connections=1)` per task and kill Celery before restarting uvicorn.
- **MongoDB path on Lightning AI**: `/data/db` is ephemeral. Always use `/teamspace/studios/this_studio/.mongodb/data`.
- **Celery queues**: All three required — `-Q scraper,replies,recommendations`. Recommendations route to their own queue to avoid blocking scraper tasks.
- **uvicorn entry point**: `uvicorn main:app` not `uvicorn app.main:app`.
