"""
app/core/config.py
==================
Centralised, type-safe configuration via pydantic-settings.

WHY THIS FILE EXISTS:
  - All environment variables are declared ONCE with types and defaults.
  - pydantic-settings reads from .env automatically (or real env vars in prod).
  - If a required variable is missing or has the wrong type, the app crashes at
    import time with a clear validation error — not randomly mid-request.
  - Every other module imports `get_settings()` — never os.getenv() directly.

DESIGN RULES:
  - No secrets are hardcoded here.
  - All fields have sensible defaults where safe; required fields have no default.
  - `get_settings()` is cached via @lru_cache so .env is read only once.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.
    Field names map directly to environment variable names (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env if present
        env_file_encoding="utf-8",
        case_sensitive=False,     # MONGODB_URI == mongodb_uri
        extra="ignore",           # Don't error on unknown env vars
    )

    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #
    app_name: str = Field(default="yt-scraper", description="Application name")
    app_version: str = Field(default="0.1.0", description="Semver release string")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment — controls behaviour guards",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level",
    )

    # ------------------------------------------------------------------ #
    # MongoDB Atlas                                                         #
    # ------------------------------------------------------------------ #
    mongodb_uri: str = Field(
        ...,   # Required — no default. App refuses to start without this.
        description="Full MongoDB Atlas connection URI (SRV format recommended)",
    )
    mongodb_db_name: str = Field(
        default="yt_scraper",
        description="Target database name inside the Atlas cluster",
    )
    mongodb_tls_allow_invalid_certs: bool = Field(
        default=False,
        description=(
            "Disable TLS certificate validation. "
            "Set true ONLY for Lightning AI dev environments where cert chain is broken. "
            "Must be false in production."
        ),
    )

    # ------------------------------------------------------------------ #
    # Redis — individual credentials                                       #
    # WHY NOT A SINGLE URL: individual fields let you rotate the password  #
    # without reconstructing multiple URLs, and each consumer (cache,      #
    # Celery broker, Celery backend) gets its own DB index automatically.  #
    # ------------------------------------------------------------------ #
    redis_host: str = Field(
        default="localhost",
        description="Redis hostname",
    )
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port",
    )
    redis_username: str = Field(
        default="default",
        description="Redis AUTH username (Redis 6+ ACL)",
    )
    redis_password: str = Field(
        default="",
        description="Redis AUTH password",
    )
    redis_ssl: bool = Field(
        default=False,
        description="Enable SSL/TLS for the Redis connection (use for Redis Cloud TLS)",
    )

    # DB index assignments — one Redis instance, three logical databases
    redis_cache_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis DB index for application cache (video meta, rate limits, sessions)",
    )
    redis_broker_db: int = Field(
        default=1,
        ge=0,
        le=15,
        description="Redis DB index for Celery task broker",
    )
    redis_result_db: int = Field(
        default=2,
        ge=0,
        le=15,
        description="Redis DB index for Celery task results",
    )

    # ------------------------------------------------------------------ #
    # Celery — computed from Redis credentials above                       #
    # These are @property (not Fields) so they don't appear in .env but   #
    # celery_app.py can still call settings.celery_broker_url unchanged.   #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Scraper (Phase 2 — declared now so config is never patched later)   #
    # ------------------------------------------------------------------ #
    scraper_max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Max simultaneous aiohttp requests per worker",
    )
    scraper_request_timeout_seconds: int = Field(
        default=30,
        ge=5,
        description="Per-request timeout in seconds",
    )
    scraper_retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retry attempts on transient HTTP failures",
    )
    scraper_retry_backoff_seconds: float = Field(
        default=2.0,
        ge=0.5,
        description="Exponential backoff base in seconds between retries",
    )

    # ------------------------------------------------------------------ #
    # Derived helpers (not env vars — computed from other fields)          #
    # ------------------------------------------------------------------ #
    @property
    def is_production(self) -> bool:
        """Guard for behaviour that must differ in prod (e.g., stricter TLS)."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    def _build_redis_url(self, db: int) -> str:
        """
        Construct a redis:// or rediss:// URL from individual credential fields.

        Format: redis://[username:password@]host:port/db
        The username is included only when a password is set — Redis Cloud
        requires `default:password@host`, plain Redis ignores the username.
        """
        scheme = "rediss" if self.redis_ssl else "redis"
        if self.redis_password:
            auth = f"{self.redis_username}:{self.redis_password}@"
        else:
            auth = ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{db}"

    @property
    def redis_url(self) -> str:
        """Connection URL for the application cache client (DB index: redis_cache_db)."""
        return self._build_redis_url(db=self.redis_cache_db)

    @property
    def celery_broker_url(self) -> str:
        """
        Celery message broker URL (DB index: redis_broker_db).

        Pool size is controlled via celery_app.conf broker_pool_limit,
        NOT via a URL query parameter (kombu rejects unknown URL params).
        """
        return self._build_redis_url(db=self.redis_broker_db)

    @property
    def celery_result_backend(self) -> str:
        """
        Celery result backend URL (DB index: redis_result_db).

        Pool size is controlled via celery_app.conf redis_max_connections.
        """
        return self._build_redis_url(db=self.redis_result_db)

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #
    @field_validator("mongodb_tls_allow_invalid_certs")
    @classmethod
    def warn_tls_in_production(cls, v: bool, info) -> bool:
        """
        We can't raise here without the full model context, but this is
        a hook for future cross-field validation.
        Actual prod guard lives in app startup (main.py lifespan).
        """
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Using @lru_cache means .env is parsed exactly once at first call.
    Call `get_settings.cache_clear()` in tests to reload settings with
    patched env vars.

    Usage:
        from app.core.config import get_settings
        settings = get_settings()
        print(settings.mongodb_db_name)
    """
    return Settings()
