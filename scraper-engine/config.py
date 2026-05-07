"""
scraper-engine/config.py
Loads all configuration from environment variables.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    FUNKER530_DIR: Path = Path(__file__).resolve().parent / "media" / "funker530"
    GEOCONFIRMED_DIR: Path = Path(__file__).resolve().parent / "media" / "geoconfirmed"
    COMBINED_DIR: Path = Path(__file__).resolve().parent / "media" / "combined"

    # ── Storage Mode (local or remote) ────────────────────────────────
    # 'local' leaves annotated files in ml-engine/media/annotated.
    # 'remote' implies upload to GCP/S3/Azure and deletes local annotated copy.
    STORAGE_MODE: str = "local" 
    REMOTE_STORAGE_BUCKET: str = "ukraine-footage-bucket"

    # ── Scraping ──────────────────────────────────────────────────────
    FUNKER530_BASE_URL: str = "https://funker530.com"
    SCRAPE_MAX_PAGES: int = 10
    SCRAPE_DELAY_SECONDS: float = 2.0
    # GeoConfirmed API
    GEOCONFIRMED_API_URL: str = "https://geoconfirmed.org/api/placemark/Ukraine"
    SCRAPE_LOOKBACK_HOURS: int = 24   # fetch everything published in the last N hours
    YTDLP_FORMAT: str = "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    def model_post_init(self, __context):
        self.FUNKER530_DIR.mkdir(parents=True, exist_ok=True)
        self.GEOCONFIRMED_DIR.mkdir(parents=True, exist_ok=True)



settings = Settings()