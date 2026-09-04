from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env(path: Path = Path(".env")) -> None:
    """Load a simple .env without overwriting process-level configuration."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    alpha_vantage_api_key: str | None
    fred_api_key: str | None
    openai_api_key: str | None
    openai_model: str
    database_path: Path
    api_token: str | None = None
    backup_directory: Path = Path("backups")
    require_api_token: bool = False
    tiingo_api_key: str | None = None
    sec_user_agent: str | None = None
    openai_research_model: str | None = None
    openai_selection_model: str | None = None
    openai_scoring_model: str | None = None
    bynara_api_key: str | None = None
    bynara_model: str = "qwen-3.8-max-free"
    bynara_base_url: str = "https://router.bynara.id/v1"
    alpaca_paper_key_id: str | None = None
    alpaca_paper_secret_key: str | None = None
    alpaca_paper_enabled: bool = False

    @classmethod
    def from_env(cls, env_file: Path = Path(".env")) -> Settings:
        load_env(env_file)
        return cls(
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            fred_api_key=os.getenv("FRED_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            database_path=Path(os.getenv("PORTFOLIOPILOT_DB_PATH", "data/portfoliopilot.db")),
            api_token=os.getenv("PORTFOLIOPILOT_API_TOKEN"),
            backup_directory=Path(os.getenv("PORTFOLIOPILOT_BACKUP_DIR", "backups")),
            require_api_token=os.getenv("PORTFOLIOPILOT_REQUIRE_API_TOKEN", "false").lower()
            in {"1", "true", "yes"},
            tiingo_api_key=os.getenv("TIINGO_API_KEY"),
            sec_user_agent=os.getenv("SEC_USER_AGENT"),
            openai_research_model=os.getenv("OPENAI_RESEARCH_MODEL") or os.getenv(
                "OPENAI_MODEL", "gpt-4o-mini",
            ),
            openai_selection_model=os.getenv("OPENAI_SELECTION_MODEL") or os.getenv(
                "OPENAI_MODEL", "gpt-4o-mini",
            ),
            openai_scoring_model=os.getenv("OPENAI_SCORING_MODEL") or os.getenv(
                "OPENAI_SELECTION_MODEL",
            ) or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            bynara_api_key=os.getenv("bynaraKey") or os.getenv("BYNARA_API_KEY"),
            bynara_model=os.getenv("BYNARA_MODEL", "qwen-3.8-max-free"),
            bynara_base_url=os.getenv("BYNARA_BASE_URL", "https://router.bynara.id/v1"),
            alpaca_paper_key_id=os.getenv("ALPACA_PAPER_KEY_ID"),
            alpaca_paper_secret_key=os.getenv("ALPACA_PAPER_SECRET_KEY"),
            alpaca_paper_enabled=os.getenv("ALPACA_PAPER_ENABLED", "false").lower()
            in {"1", "true", "yes"},
        )

    def validate_alpaca_paper(self) -> None:
        if not self.alpaca_paper_enabled:
            raise ValueError("Alpaca paper execution is disabled")
        if not self.alpaca_paper_key_id or not self.alpaca_paper_secret_key:
            raise ValueError("ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY are required")

    def validate_worker(self) -> None:
        if not self.alpha_vantage_api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required by the market snapshot worker")
        if not self.database_path:
            raise ValueError("PORTFOLIOPILOT_DB_PATH is required")

    def validate_exposed_api(self) -> None:
        if not self.api_token or len(self.api_token) < 32:
            raise ValueError("an API token of at least 32 characters is required for an exposed API")
