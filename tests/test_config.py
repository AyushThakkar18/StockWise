from pathlib import Path

import pytest

from portfoliopilot.config import Settings


def test_worker_requires_market_data_key() -> None:
    settings = Settings(None, None, None, "gpt-4o-mini", Path("data/test.db"))
    with pytest.raises(ValueError, match="ALPHA_VANTAGE"):
        settings.validate_worker()


def test_worker_configuration_accepts_required_key() -> None:
    Settings("test-key", None, None, "gpt-4o-mini", Path("data/test.db")).validate_worker()


def test_exposed_api_requires_long_token() -> None:
    settings = Settings("key", None, None, "gpt-4o-mini", Path("data/test.db"), "short")
    with pytest.raises(ValueError, match="32"):
        settings.validate_exposed_api()
