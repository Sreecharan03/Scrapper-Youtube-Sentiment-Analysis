"""
app/scraper/constants.py
=========================
YouTube internal API constants and default HTTP headers.

NOTE ON THE API KEY:
  AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8 is the well-known public
  Innertube API key for the WEB client.  It has been stable for years.
  We also extract it from the live page on every initial fetch so the
  scraper automatically picks up changes without a code deploy.

NOTE ON CLIENT VERSION:
  Extracted fresh from every initial page load.  The fallback constant
  here is used only when extraction fails.
"""

# ── Endpoints ──────────────────────────────────────────────────────────────
YT_BASE_URL        = "https://www.youtube.com"
YT_WATCH_URL       = "https://www.youtube.com/watch?v={video_id}"
YT_INNERTUBE_URL   = "https://www.youtube.com/youtubei/v1/next"

# ── Innertube client identity ──────────────────────────────────────────────
YT_API_KEY_FALLBACK      = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
YT_CLIENT_NAME           = "WEB"
YT_CLIENT_NAME_INT       = "1"                     # numeric form for X-YouTube-Client-Name
YT_CLIENT_VERSION_FALLBACK = "2.20240601.05.00"

# ── Default HTTP headers ───────────────────────────────────────────────────
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language":    "en-US,en;q=0.9",
    "Accept-Encoding":    "gzip, deflate, br",
    "Accept":             "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "DNT":                "1",
    "Sec-Fetch-Site":     "same-origin",
    "Sec-Fetch-Mode":     "navigate",
    "Sec-Fetch-Dest":     "document",
}

INNERTUBE_POST_HEADERS = {
    "Content-Type":            "application/json",
    "X-YouTube-Client-Name":   YT_CLIENT_NAME_INT,
    "X-YouTube-Client-Version": YT_CLIENT_VERSION_FALLBACK,
    "Origin":                  "https://www.youtube.com",
}

# ── Scraping behaviour ─────────────────────────────────────────────────────
# Random delay range between API calls (milliseconds).
# TLC scraping uses this range.  Reply scraping skips the delay entirely
# (each reply chain uses a unique token — not repetitive on the same endpoint).
REQUEST_DELAY_MIN_MS  = 50
REQUEST_DELAY_MAX_MS  = 100

# HTTP timeouts (seconds)
CONNECT_TIMEOUT = 15
READ_TIMEOUT    = 30

# Retry limits
MAX_API_RETRIES     = 3
MAX_REPLY_RETRIES   = 3

# Backoff delays for rate limiting (seconds)
RATE_LIMIT_BACKOFF  = [60, 120, 240]

# Sub-batch / batch sizes
SUB_BATCH_SIZE       = 1000   # comments per MongoDB write
BATCH_SIZE           = 5_000  # TLCs per Celery TLC batch task
REPLY_TASK_BATCH_SIZE = 25    # reply chains processed per Celery reply task

# YouTube comments per API response page (internal API always returns ~20)
YT_PAGE_SIZE = 20

# Token expiry safety margin — refresh before 6-hour YouTube expiry
TOKEN_MAX_AGE_SECONDS = 5 * 60 * 60  # 5 hours (refresh before 6-hour limit)

# ── Error signal patterns (substrings to match in YouTube responses) ───────
TOKEN_EXPIRED_SIGNALS = [
    "invalidationId",
    "INVALID_TOKEN",
    "continuation token",
]
RATE_LIMIT_STATUS_CODES = {429, 503}
PERMANENT_ERROR_CODES   = {404, 410}   # video not found / removed
