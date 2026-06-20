# YouTube Audience Intelligence Platform

Scrapes every comment (top-level + replies) from any YouTube video using YouTube's internal Innertube API — no API key, no quota limits — then runs a full NLP + LLM analysis pipeline to surface audience intent, topic clusters, content gaps, misconceptions, and actionable recommendations for educational creators.

## Features

- **Zero-quota scraping** — YouTube Innertube `/youtubei/v1/next` API; handles 10K+ comment threads
- **Multi-language transcripts** — fetches captions via `youtube-transcript-api` (manual → auto → any)
- **LLM video summary** — two-call pipeline (draft → self-critique) with prompt caching; extracts key topics, claims, controversy triggers, video promises
- **Comment classification** — Groq `llama-3.1-8b-instant`; 8 intent labels + sentiment + `answered_by_video` flag
- **Preprocessing pipeline** — emoji → Unicode, pure-number filter, punctuation normalization, reply relevance filter
- **BERTopic clustering** — UMAP + HDBSCAN + CountVectorizer; auto-adjusts cluster size, reduces outliers, labels clusters via Groq few-shot prompting
- **Content gap detection** — cosine similarity of cluster labels vs video key topics/claims; surfaces uncovered audience demand
- **Audience recommendations** — ranked content gaps, misconception map, controversy hotspots, top unanswered questions; enriched with chain-of-thought LLM analysis
- **Per-intent audience summaries** — one LLM call generates 2-3 sentence creator-ready summaries for all 8 intent categories + overall video summary; cached in MongoDB
- **Auto-retry** — up to 3 automatic retry rounds on classification failures
- **Async + distributed** — FastAPI + Celery (two queues) + local MongoDB + Redis

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Task queue | Celery (`scraper` + `replies` queues) |
| Database | MongoDB (local, persistent path) |
| Cache / broker | Redis |
| Scraping | aiohttp + YouTube Innertube API |
| Transcript | `youtube-transcript-api` v1.x |
| LLM summary & intent summaries | Anthropic Claude Haiku |
| Comment classification & cluster labeling | Groq `llama-3.1-8b-instant` |
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
Phase 3D — Gap analysis + recommendations       ✅ Complete
Phase 3E — Per-intent audience summaries        ✅ Complete
Phase 3F — Dashboard API endpoints              🔜 Next
Phase 3G — React / Next.js dashboard            🔜 Planned
```

### Results (eye health video — 3,569 comments)

| Metric | Value |
|---|---|
| Total comments scraped | 3,569 |
| Classified | 3,092 (86.6%) |
| Topic clusters found | 13 |
| Content gaps detected | 1 (Astigmatism — 76% questions, 50 comments) |
| Controversy hotspots | 1 (LED Safety — 73.1% criticism, 49 comments) |
| Misconception clusters | 5 |
| Top intent | Praise 41.7% |
| Positive sentiment | 47.4% |

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
│       │   └── intent_summaries.py      # per-intent audience summaries (3E)
│       └── schemas/
├── core/
│   ├── config.py
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
│   ├── classifier.py                    # Groq batched comment classifier
│   ├── clustering_service.py            # BERTopic pipeline (3C)
│   ├── recommendation_service.py        # gap analysis engine (3D)
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
        ├── clustering_tasks.py          # cluster_comments (3C)
        ├── recommendation_tasks.py      # generate_recommendations (3D)
        └── intent_summary_tasks.py      # generate_intent_summaries (3E)
```

---

## Setup

### Prerequisites

- Python 3.12 (conda env: `cloudspace`)
- MongoDB (local)
- Redis

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in:

```env
# MongoDB (local)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=yt_scraper

# Redis
REDIS_HOST=your-redis-host
REDIS_PORT=6379

# Anthropic (Phase 3A summary + Phase 3E intent summaries)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# Groq (Phase 3B classification + Phase 3C cluster labeling)
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant
```

---

## Running

### 1. Start MongoDB (persistent path — survives restarts)

```bash
mongod --dbpath /teamspace/studios/this_studio/.mongodb/data \
       --logpath /teamspace/studios/this_studio/.mongodb/mongod.log \
       --fork
```

### 2. Start Celery workers

Both queues required — `replies` handles reply-batch tasks:

```bash
python3 -m celery -A app.workers.celery_app worker \
       -Q scraper,replies \
       --concurrency=4 \
       --loglevel=INFO
```

### 3. Start FastAPI

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### Scraping

```
POST /api/v1/jobs                              Submit scrape job for a video URL
GET  /api/v1/jobs/{job_id}                     Job status + progress
GET  /api/v1/comments?video_id=...             List scraped comments
GET  /api/v1/videos/{video_id}                 Video metadata
```

### Phase 3A — Transcript + Summary

```
POST /api/v1/transcripts/{video_id}            Fetch transcript (multi-language)
GET  /api/v1/transcripts/{video_id}            Get stored transcript
POST /api/v1/summaries/{video_id}              Generate LLM video summary
GET  /api/v1/summaries/{video_id}              Get stored summary
```

### Phase 3B — Classification

```
POST /api/v1/analysis/{video_id}/classify      Classify all comments (auto-retries 3x)
POST /api/v1/analysis/{video_id}/classify/retry  Retry only failed comments
GET  /api/v1/analysis/{video_id}               Get intent + sentiment breakdown
```

### Phase 3C — Topic Clustering

```
POST /api/v1/clusters/{video_id}               Run BERTopic clustering
GET  /api/v1/clusters/{video_id}/status        Clustering job status
GET  /api/v1/clusters/{video_id}               All clusters sorted by size
GET  /api/v1/clusters/{video_id}/{cluster_id}  Single cluster detail
```

### Phase 3D — Recommendations

```
POST /api/v1/recommendations/{video_id}        Generate audience recommendations
GET  /api/v1/recommendations/{video_id}/status Job status
GET  /api/v1/recommendations/{video_id}        Full recommendations result
```

### Phase 3E — Intent Summaries

```
POST /api/v1/intent-summaries/{video_id}        Generate per-intent summaries (cached)
GET  /api/v1/intent-summaries/{video_id}/status Job status
GET  /api/v1/intent-summaries/{video_id}        All 8 intent summaries + overall
```

---

### Full pipeline for a new video

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

# 6. Fetch results
curl $BASE/api/v1/recommendations/$VIDEO_ID | python3 -m json.tool
curl $BASE/api/v1/intent-summaries/$VIDEO_ID | python3 -m json.tool
```

---

## Analysis Pipeline

### Phase 3C — BERTopic Clustering

```
Filter + dedup (spam, len<20, text_hash dedup)
  ↓
Embed (all-MiniLM-L6-v2)
  ↓
BERTopic: UMAP(n_components=5, cosine) + HDBSCAN(min_size=dynamic) + CountVectorizer(stop_words=english, ngram=(1,2))
  ↓
Auto-adjust: outlier_ratio > 35% → reduce min_cluster_size × 0.7, refit
  ↓
reduce_outliers(strategy=embeddings)   ← NEVER call update_topics() after this
  ↓
Groq few-shot labeling (5 domain examples + video context, temp=0.2)
  ↓
Gap detection: cosine_sim(cluster_label, key_topics + key_claims) < 0.35 → is_content_gap=True
```

### Phase 3D — Recommendations

Four analysis types, all from existing MongoDB data — no new models:

| Type | Signal | Filter |
|---|---|---|
| Content gaps | `is_content_gap=True` OR `question_pct > 40% AND gap_sim < 0.42` | Skip fan clusters (praise > 50%) |
| Misconceptions | misconception-labeled comments grouped by cluster | Skip fan clusters, min text length 30 chars |
| Controversy hotspots | `criticism_pct > 25%` AND dominant sentiment ≠ positive | Match against `summary.controversy_triggers` via embedding |
| Unanswered questions | `answered_by_video=False`, sorted by like_count | Skip fan clusters |

All findings enriched with one Groq call using expert persona + chain-of-thought + few-shot examples.

### Phase 3E — Intent Summaries

One LLM call per video generates:
- **8 intent summaries** (question, praise, criticism, confusion, misconception, request — via LLM; spam + off_topic — static count lines)
- **1 overall summary** — 3-sentence arc of dominant feeling, #1 audience need, and risk
- Cached in MongoDB — re-triggers only if comment count grows > 10%

---

## MongoDB Collections

| Collection | Purpose |
|---|---|
| `comments` | All TLCs and replies, flat structure |
| `jobs` | One doc per scrape job with progress tracking |
| `scrape_batches` | Per-batch checkpoint state |
| `scrape_sessions` | Resume tokens for interrupted scrapes |
| `comment_history` | Edit detection archive |
| `failed_replies` | Reply batches that exhausted all retries |
| `transcripts` | Video transcripts (all languages) |
| `summaries` | LLM-generated video summaries |
| `comment_analysis` | Aggregate classification stats per video |
| `reports` | Clean client-facing export per video |
| `clusters` | One doc per topic cluster (3C) |
| `cluster_info` | Clustering job state + metadata (3C) |
| `recommendations` | Gap analysis + recommendation results (3D) |
| `intent_summaries` | Per-intent LLM summaries, cached (3E) |

---

## Key Engineering Notes

- **YouTube scraping**: Uses Innertube `/youtubei/v1/next` with ViewModel+mutations format (2024+). The old `commentRenderer` format is legacy and misses most comments on modern videos. Install `Brotli` — YouTube returns brotli-encoded responses.
- **BERTopic keyword quality**: Never call `update_topics()` after `reduce_outliers()`. It forces a full c-TF-IDF recompute on enlarged clusters, causing stop words to dominate keywords. Preserve the original `fit_transform` representations.
- **numpy → MongoDB**: BERTopic returns `numpy.int64` / `numpy.float64`. PyMongo rejects them — wrap every BERTopic output with `int()` / `float()` before storing.
- **Fan cluster filtering**: BERTopic may type fan praise clusters as "topic". Filter any cluster with praise > 50% from content gap, misconception, and unanswered question analysis in the recommendation layer — do not re-cluster.
- **Groq 8b + JSON**: Use `response_format={"type": "json_object"}` — the model wraps output in markdown fences without it. Wrap output schema in a root object, not a bare array.
- **Managed Redis**: Only DB 0 is available on Redis Cloud free tier; max 30 connections. Use `BlockingConnectionPool(max_connections=1)` per Celery task.
- **MongoDB path on Lightning AI**: `/data/db` is ephemeral and wiped on restart. Always use `/teamspace/studios/this_studio/.mongodb/data`.
- **Celery queues**: Both `-Q scraper,replies` are required. Reply batch tasks route to `replies` queue — starting only `-Q scraper` leaves all reply fetching stuck at 0%.
- **uvicorn entry point**: `main.py` is at project root. Use `uvicorn main:app`, not `uvicorn app.main:app`.
