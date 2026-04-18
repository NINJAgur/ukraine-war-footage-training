"""
scraper-engine/config.py
Loads all configuration from environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────
    # Sync URL used by Celery tasks (SQLAlchemy sync engine)
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@localhost:5432/ukraine_footage"

    # ── Redis / Celery ────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Media Storage — absolute paths inside scraper-engine/ ────────
    MEDIA_ROOT: Path = Path(__file__).resolve().parent / "media"
    RAW_VIDEO_DIR: Path = Path(__file__).resolve().parent / "media" / "raw"

    # ── Scraping ──────────────────────────────────────────────────────
    FUNKER530_BASE_URL: str = "https://funker530.com"
    SCRAPE_MAX_PAGES: int = 10
    SCRAPE_DELAY_SECONDS: float = 2.0
    FUNKER530_MAX_POSTS: int = 20   # max Ukraine posts to process per run
    # GeoConfirmed API
    GEOCONFIRMED_API_URL: str = "https://geoconfirmed.org/api/placemark/Ukraine"
    GEOCONFIRMED_MAX_INCIDENTS: int = 20   # max video incidents to process per run
    YTDLP_FORMAT: str = "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    # ── Kaggle ────────────────────────────────────────────────────────
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    # Comma-separated Kaggle dataset slugs
    KAGGLE_BASELINE_DATASETS: str = "sudipchakrabarty/kiit-mita"
    DATASETS_DIR: Path = Path(__file__).resolve().parent / "media" / "datasets"

    def model_post_init(self, __context):
        # Ensure media directories exist
        self.RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def kaggle_dataset_list(self) -> list[str]:
        return [d.strip() for d in self.KAGGLE_BASELINE_DATASETS.split(",") if d.strip()]


settings = Settings()
