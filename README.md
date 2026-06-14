# YouTube Audience Intelligence Platform

Scrapes every comment (top-level + replies) from any YouTube video using YouTube's internal Innertube API — no API key, no quota limits — then runs a full NLP analysis pipeline to surface audience intent, sentiment, content gaps, and misconceptions for educational creators.

## Features

- **Zero-quota scraping** — YouTube Innertube `/youtubei/v1/next` API; handles 10K+ comment threads
- **Multi-language transcripts** — fetches captions via `youtube-transcript-api` (manual → auto → any)
- **LLM video summary** — Claude Haiku two-call pipeline (draft → self-critique) with prompt caching
- **Comment classification** — Groq `llama-3.1-8b-instant`; 7 intent labels + sentiment + content gap detection
- **Preprocessing pipeline** — emoji → Unicode, pure-number filter, punctuation normalization
- **Reply relevance filter** — `all-MiniLM-L6-v2` embeddings; removes off-topic reply threads before LLM
- **Auto-retry** — up to 3 automatic retry rounds on classification failures, no manual intervention
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
| LLM summary | Anthropic Claude `claude-haiku-4-5-20251001` |
| Classification | Groq `llama-3.1-8b-instant` via OpenAI-compatible API |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim) |

---

## Project Status

```
Phase 1 — Architecture & infrastructure     ✅ Complete
Phase 2 — Comment scraping pipeline         ✅ Complete
Phase 3A — Transcript + LLM summary         ✅ Complete
Phase 3B — Comment classification           ✅ Complete
Phase 3C — BERTopic topic clustering        🔜 Next
Phase 3D — Gap analysis                     🔜 Planned
Phase 3E — Dashboard API endpoints          🔜 Planned
Phase 3F — React / Next.js dashboard        🔜 Planned
```

### Phase 3B Results (eye health video — 3,569 comments)

| Metric | Value |
|---|---|
| Total comments | 3,569 |
| Classified | 3,092 (86.6%) |
| Skipped (non-English / numbers) | 473 |
| Permanent failures | 4 |
| Top intent | Praise 41.7% |
| Positive sentiment | 47.4% |

---

## Project Structure

```
app/
├── api/
│   ├── router.py                        # mounts all v1 routers
│   └── v1/
│       ├── endpoints/
│       │   ├── jobs.py                  # scrape job submit/status
│       │   ├── comments.py              # comment listing
│       │   ├── videos.py                # video metadata
│       │   ├── transcripts.py           # transcript fetch/get
│       │   ├── summaries.py             # LLM summary generate/get
│       │   └── analysis.py              # classification + retry + results
│       └── schemas/
│           └── analysis.py              # Pydantic response models
├── core/
│   ├── config.py                        # pydantic-settings (.env → typed config)
│   └── logging.py                       # structlog setup
├── db/
│   ├── init_db.py                       # collection + index creation
│   └── repositories/
│       ├── comment_repo.py              # CRUD for comments collection
│       └── classification_repo.py       # comment_analysis + reports collections
├── scraper/
│   └── pipeline.py                      # Innertube scrape logic
├── services/
│   ├── classifier.py                    # CommentClassifier (Groq batched async)
│   ├── text_preprocessor.py             # emoji replace, number filter, punctuation clean
│   ├── relevance_filter.py              # all-MiniLM-L6-v2 reply relevance filter
│   └── emoji_dict.json                  # 1,918 emoji → Unicode description map
└── workers/
    ├── celery_app.py                    # Celery app + task routing
    └── tasks/
        ├── scraper_tasks.py             # TLC scrape, reply batch tasks
        ├── summary_tasks.py             # transcript + LLM summary tasks
        └── classification_tasks.py      # classify_comments with auto-retry
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
REDIS_HOST=localhost
REDIS_PORT=6379

# Anthropic (Phase 3A — LLM summary)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# Groq (Phase 3B — comment classification)
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

### 2. Start Redis

```bash
redis-server --daemonize yes
```

### 3. Start Celery workers

Both queues are required — `replies` handles reply-batch tasks:

```bash
celery -A app.workers.celery_app worker \
       -Q scraper,replies \
       --concurrency=4 \
       --loglevel=INFO
```

### 4. Start FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### Scraping

```
POST /api/v1/jobs                          Submit a scrape job for a video URL
GET  /api/v1/jobs/{job_id}                 Job status + progress
GET  /api/v1/comments?video_id=...         List scraped comments
GET  /api/v1/videos/{video_id}             Video metadata
```

### Analysis

```
POST /api/v1/transcripts/{video_id}        Fetch transcript (multi-language)
GET  /api/v1/transcripts/{video_id}        Get stored transcript

POST /api/v1/summaries/{video_id}          Generate LLM summary (Claude Haiku)
GET  /api/v1/summaries/{video_id}          Get stored summary

POST /api/v1/analysis/{video_id}/classify         Classify all comments (auto-retries 3x)
POST /api/v1/analysis/{video_id}/classify/retry   Retry only failed comments
GET  /api/v1/analysis/{video_id}                  Get classification results + intent breakdown
```

### Full pipeline for a new video

```bash
VIDEO_ID="nCnZX8zs4LI"
BASE="http://localhost:8000"

# 1. Scrape
curl -X POST $BASE/api/v1/jobs \
     -H "Content-Type: application/json" \
     -d '{"video_url": "https://www.youtube.com/watch?v='$VIDEO_ID'"}'

# 2. Transcript + summary (run after scrape completes)
curl -X POST $BASE/api/v1/transcripts/$VIDEO_ID
curl -X POST $BASE/api/v1/summaries/$VIDEO_ID

# 3. Classify comments (run after summary completes)
curl -X POST $BASE/api/v1/analysis/$VIDEO_ID/classify

# 4. Poll results
curl $BASE/api/v1/analysis/$VIDEO_ID
```

---

## Classification Pipeline (Phase 3B)

### Preprocessing (before every LLM batch)

1. **Non-English filter** — skips comments with non-Latin Unicode characters
2. **Emoji → Unicode** — 1,918 emojis replaced with `[name]` descriptions (e.g. 💀 → `[skull]`) so the LLM understands Gen Z sentiment correctly
3. **Pure number skip** — comments that are only digits/formatting are skipped
4. **Punctuation clean** — collapses `....` → `.`, removes markdown noise (`***`, `###`)
5. **Reply relevance filter** — `all-MiniLM-L6-v2` cosine similarity; reply kept if:
   - `reply_video_sim ≥ 0.25` (reply directly relates to video), OR
   - `reply_parent_sim ≥ 0.45` AND `parent_video_sim ≥ 0.25` (on-topic thread)

### Intent labels (multi-label per comment)

`question` · `praise` · `criticism` · `confusion` · `misconception` · `request` · `spam` · `off_topic`

### Context injection

The video's LLM summary (`key_topics`, `key_claims`, `controversy_triggers`) is injected into the system prompt so the model detects misconceptions against what the video *actually* claimed.

### Auto-retry

After initial classification, if any comments failed, the task self-chains up to `MAX_AUTO_RETRIES = 3` times targeting only failed comments. Remaining failures after 3 rounds are accepted as permanent.

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
| `reports` | Clean client-facing export per video (upserted) |

---

## Key Engineering Notes

- **YouTube scraping**: Uses the Innertube `/youtubei/v1/next` API with ViewModel+mutations format (2024+). The old `commentRenderer` format is legacy and will miss most comments on modern videos.
- **Groq 8b + JSON**: Must use `response_format={"type": "json_object"}` — the model wraps output in markdown fences without it. Output schema must be a wrapper object (`{"classifications": [...]}`) not a bare array.
- **Managed Redis**: Only DB 0 is available on Redis Cloud free tier; max 30 connections. The app uses three logical DB indices for cache/broker/results.
- **MongoDB path**: On Lightning AI, `/data/db` is ephemeral and wiped on restart. Always use a path inside the persistent workspace.
- **Celery queues**: Both `-Q scraper,replies` are required. Reply batch tasks are routed to the `replies` queue; starting only `-Q scraper` leaves all reply fetching stuck at 0%.
