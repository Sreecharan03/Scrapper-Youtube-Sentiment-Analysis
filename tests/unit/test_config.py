"""
tests/unit/test_config.py
==========================
Unit tests for the configuration module.

These tests verify:
  - Settings load correctly from env vars
  - Validation rejects invalid values
  - Derived properties work correctly
  - lru_cache can be cleared between tests

MARKER: unit — no I/O, no network, no DB.
"""

import os

import pytest

from app.core.config import Settings, get_settings


@pytest.mark.unit
class TestSettingsLoading:

    def setup_method(self):
        """Clear the lru_cache before each test so env changes take effect."""
        get_settings.cache_clear()

    def test_defaults_are_sane(self):
        """Settings should have reasonable defaults for non-required fields."""
        s = Settings(
            mongodb_uri="mongodb://localhost:27017",  # Required field — provide it
        )
        assert s.app_name == "yt-scraper"
        assert s.app_env == "development"
        assert s.log_level == "INFO"
        assert s.mongodb_db_name == "yt_scraper"
        assert s.mongodb_tls_allow_invalid_certs is False
        assert s.scraper_max_concurrent_requests == 5

    def test_mongodb_uri_is_required(self):
        """Missing MONGODB_URI must raise at instantiation — not silently use None."""
        with pytest.raises(Exception):  # pydantic ValidationError
            Settings()  # No MONGODB_URI in env and no default

    def test_invalid_app_env_rejected(self):
        """app_env only accepts: development | staging | production."""
        with pytest.raises(Exception):
            Settings(mongodb_uri="mongodb://localhost", app_env="prod")  # Invalid literal

    def test_invalid_log_level_rejected(self):
        """log_level must be a valid Python log level."""
        with pytest.raises(Exception):
            Settings(mongodb_uri="mongodb://localhost", log_level="VERBOSE")

    def test_scraper_concurrency_bounds(self):
        """scraper_max_concurrent_requests must be between 1 and 50."""
        with pytest.raises(Exception):
            Settings(mongodb_uri="mongodb://localhost", scraper_max_concurrent_requests=0)
        with pytest.raises(Exception):
            Settings(mongodb_uri="mongodb://localhost", scraper_max_concurrent_requests=51)

    def test_is_production_property(self):
        s = Settings(mongodb_uri="mongodb://localhost", app_env="production")
        assert s.is_production is True
        assert s.is_development is False

    def test_is_development_property(self):
        s = Settings(mongodb_uri="mongodb://localhost", app_env="development")
        assert s.is_development is True
        assert s.is_production is False

    def test_get_settings_is_cached(self):
        """get_settings() must return the same object on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_reloads_settings(self, monkeypatch):
        """After cache_clear(), a changed env var is picked up."""
        s1 = get_settings()
        get_settings.cache_clear()
        monkeypatch.setenv("APP_NAME", "patched-name")
        s2 = get_settings()
        assert s2.app_name == "patched-name"
        # Restore for other tests
        get_settings.cache_clear()
