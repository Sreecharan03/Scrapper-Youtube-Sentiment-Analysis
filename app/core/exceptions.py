"""
app/core/exceptions.py
======================
Custom exception hierarchy for the entire application.

WHY THIS FILE EXISTS:
  - A flat `Exception` tells you nothing. A `ScraperRateLimitError` tells you
    exactly what happened and how to respond to it.
  - FastAPI exception handlers (in main.py) map these classes to HTTP status
    codes — so the same exception type always returns the same HTTP response.
  - Celery task retry logic checks `isinstance(exc, RetryableError)` to decide
    whether to retry or fail permanently.

HIERARCHY:
  AppBaseError                     ← root of all our custom exceptions
  ├── ConfigurationError           ← startup/config failures (fatal)
  ├── DatabaseError                ← MongoDB failures
  │   ├── DatabaseConnectionError  ← can't reach Atlas
  │   └── DatabaseOperationError   ← query/write failure
  ├── ScraperError                 ← scraping failures
  │   ├── ScraperRateLimitError    ← 429 / quota — backoff + retry
  │   ├── ScraperAuthError         ← YouTube blocked the session — need new cookies
  │   ├── ScraperParseError        ← response shape changed — needs dev fix
  │   └── ScraperTimeoutError      ← request timed out — retry
  ├── JobError                     ← Celery job management failures
  │   ├── JobNotFoundError         ← queried job doesn't exist
  │   └── JobAlreadyExistsError    ← duplicate job submission
  └── ValidationError              ← bad caller input (maps to HTTP 422)
"""


# ============================================================
# Base
# ============================================================

class AppBaseError(Exception):
    """
    Root of all application-defined exceptions.
    Never catch this directly in business logic — catch specific subclasses.
    """

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail  # Extra context for logs, not exposed to API callers

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, detail={self.detail!r})"


class RetryableError(AppBaseError):
    """
    Mixin base for errors that Celery workers SHOULD retry.
    Usage: `class ScraperRateLimitError(RetryableError, ScraperError): ...`
    Celery tasks check: `isinstance(exc, RetryableError)`
    """
    pass


# ============================================================
# Configuration
# ============================================================

class ConfigurationError(AppBaseError):
    """
    Raised during startup if required config is missing or invalid.
    This is always fatal — the app should not start.
    """
    pass


# ============================================================
# Database
# ============================================================

class DatabaseError(AppBaseError):
    """Base for all MongoDB-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """
    Cannot establish or maintain a connection to MongoDB Atlas.
    Usually caused by bad URI, network issues, or Atlas IP allowlist.
    Retryable at the infrastructure level.
    """
    pass


class DatabaseOperationError(DatabaseError):
    """
    A database query or write operation failed after connection was established.
    Includes write conflicts, validation errors at DB level, timeouts.
    """
    pass


# ============================================================
# Scraper
# ============================================================

class ScraperError(AppBaseError):
    """Base for all scraping-related errors."""
    pass


class ScraperRateLimitError(RetryableError, ScraperError):
    """
    YouTube returned 429 or a continuation token indicating rate limiting.
    Celery should retry with exponential backoff.
    """
    pass


class ScraperAuthError(ScraperError):
    """
    YouTube blocked the request session (bad/expired cookies, bot detection).
    NOT retryable without fresh credentials — requires human intervention.
    """
    pass


class ScraperParseError(ScraperError):
    """
    YouTube's internal API response shape changed and our parser broke.
    NOT retryable — requires a code fix.
    Include the raw response fragment in `detail` for debugging.
    """
    pass


class ScraperTimeoutError(RetryableError, ScraperError):
    """
    aiohttp request timed out waiting for YouTube.
    Retryable with backoff.
    """
    pass


class ScraperVideoNotFoundError(ScraperError):
    """
    The target video does not exist, is private, or has been deleted.
    NOT retryable — job should be marked as permanently failed.
    """
    pass


# ============================================================
# Jobs
# ============================================================

class JobError(AppBaseError):
    """Base for Celery job management errors."""
    pass


class JobNotFoundError(JobError):
    """Queried job ID does not exist in MongoDB."""
    pass


class JobAlreadyExistsError(JobError):
    """
    A scrape job for this video_id is already queued or running.
    Prevents duplicate work.
    """
    pass


# ============================================================
# Validation
# ============================================================

class AppValidationError(AppBaseError):
    """
    Caller supplied invalid input that passed Pydantic schema but failed
    business-rule validation (e.g., video_id format wrong for YT).
    Maps to HTTP 422 in FastAPI exception handlers.
    """
    pass
