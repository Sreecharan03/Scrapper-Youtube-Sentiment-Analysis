"""
app/core/logging.py
===================
Structured logging setup using structlog.

WHY THIS FILE EXISTS:
  - Plain print() / logging.info() produce unstructured text that can't be
    queried, filtered, or alerted on in any modern log platform.
  - structlog produces JSON like:
      {"event": "db_connected", "db": "yt_scraper", "level": "info", "timestamp": "..."}
    Every field is queryable. You can filter by level, module, job_id, video_id.
  - In development: human-readable coloured console output.
  - In production: strict JSON for log aggregators.

USAGE:
  from app.core.logging import get_logger

  logger = get_logger(__name__)
  logger.info("job_started", job_id=str(job_id), video_id=video_id)
  logger.error("scrape_failed", exc_info=True, video_id=video_id)

DO NOT:
  - Use logging.getLogger() directly — always use get_logger() from this module.
  - Log secrets (URIs, tokens, passwords). Log IDs and status codes only.
"""

import logging
import sys
from typing import Any

import structlog

from app.core.config import get_settings


def _get_renderer() -> Any:
    """
    Return the appropriate renderer based on environment.
    Dev: coloured, human-readable console output.
    Prod: strict JSON for log aggregators (Datadog, CloudWatch, etc.).
    """
    settings = get_settings()

    if settings.is_production:
        return structlog.processors.JSONRenderer()
    else:
        return structlog.dev.ConsoleRenderer(colors=True)


def setup_logging() -> None:
    """
    Configure structlog and the stdlib logging root logger.
    Call this ONCE at application startup (in main.py lifespan).

    After this call, all loggers — structlog and stdlib — output through
    the same pipeline, so you get consistent formatting everywhere.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # --- Configure stdlib root logger ---
    # structlog wraps stdlib, so configuring the root captures everything
    # including third-party libraries (uvicorn, motor, celery).
    logging.basicConfig(
        format="%(message)s",   # structlog handles the actual formatting
        stream=sys.stdout,
        level=log_level,
        force=True,             # Override any previous basicConfig calls
    )

    # Quieten noisy third-party loggers in development
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    # --- Configure structlog pipeline ---
    shared_processors: list[Any] = [
        # Add log level name to every event
        structlog.stdlib.add_log_level,
        # Add logger name (usually __name__ of the calling module)
        structlog.stdlib.add_logger_name,
        # Add ISO 8601 timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Render exceptions with full tracebacks
        structlog.processors.ExceptionRenderer(),
        # Allow positional format strings: log.info("user %s logged in", user_id)
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Render stack_info if provided
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors + [
            # Bridge from structlog to stdlib before rendering
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        # Use stdlib LoggerFactory so `add_logger_name` can read logger.name
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Attach structlog's ProcessorFormatter to the stdlib root handler
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=_get_renderer(),
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a bound structlog logger for a given module.

    Usage:
        logger = get_logger(__name__)
        logger.info("event_name", key="value", another_key=123)

    The `name` is automatically included in every log line as `logger=name`.
    """
    return structlog.get_logger(name)
