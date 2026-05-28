"""
training-engine/config.py
Loads all training-engine configuration from environment variables.
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `shared` package is importable
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@localhost:5432/ukraine_footage"

    # ── Redis / Celery ────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Media Storage — all paths anchored to repo root ──────────────
    MEDIA_ROOT: Path = Path(__file__).parent.parent / "media"
    RAW_VIDEO_DIR: Path = Path(__file__).parent.parent / "media" / "raw"
    ANNOTATED_VIDEO_DIR: Path = Path(__file__).parent.parent / "inference-engine" / "media"

    # ── Storage Mode (local or remote) ────────────────────────────────
    STORAGE_MODE: str = "local"
    REMOTE_STORAGE_BUCKET: str = "ukraine-footage-bucket"

    # ── Training Runs — under training-engine/ ────────────────────────
    RUNS_DIR: Path = Path(__file__).parent / "runs"

    # ── Scraped datasets (written by inference-engine) ───────────────
    DATASETS_DIR: Path = Path(__file__).parent.parent / "inference-engine" / "media" / "scraped_datasets"

    # ── Kaggle dataset cache — kept inside the project ────────────────
    KAGGLE_CACHE_DIR: Path = Path(__file__).parent / "media" / "kaggle_datasets"

    # ── GPU / Model ───────────────────────────────────────────────────
    GPU_DEVICE: str = "cuda:0"          # device for training and inference
    YOLO_MODEL: str = "yolov8m.pt"      # base model for Stage 1 baseline
    YOLO_BATCH_SIZE: int = 8            # max for 8GB VRAM with yolov8m
    YOLO_IMG_SIZE: int = 640
    YOLO_EPOCHS_BASELINE: int = 10
    YOLO_EPOCHS_FINETUNE: int = 10
    YOLO_FINETUNE_MAX_CYCLES: int = 4   # baseline(10) + 4×finetune(10) = 50 total epochs

    # ── Multi-model class definitions ─────────────────────────────────
    # Universal 3-class vocabulary — all models share the same class IDs.
    # 0=AIRCRAFT  1=VEHICLE  2=PERSONNEL
    MODEL_CLASSES: dict = {
        "GENERAL":   ["aircraft", "vehicle", "personnel"],
        "AIRCRAFT":  ["aircraft", "vehicle", "personnel"],
        "VEHICLE":   ["aircraft", "vehicle", "personnel"],
        "PERSONNEL": ["aircraft", "vehicle", "personnel"],
    }
    # Render colours per model type (BGR for OpenCV) — matched to CSS oklch vars in style.css
    MODEL_COLORS: dict = {
        "GENERAL":   (  0, 105, 223),  # BGR ← oklch(0.65 0.18 55)  rgb(223,105,0)
        "AIRCRAFT":  (200, 153,   0),  # BGR ← oklch(0.62 0.16 220) rgb(0,153,200)
        "VEHICLE":   ( 61,  59, 222),  # BGR ← oklch(0.60 0.20 25)  rgb(222,59,61)
        "PERSONNEL": ( 48, 154,  24),  # BGR ← oklch(0.60 0.18 145) rgb(24,154,48)
    }

    def model_post_init(self, __context):
        for d in [
            self.ANNOTATED_VIDEO_DIR,
            self.RUNS_DIR,
            self.KAGGLE_CACHE_DIR,
            self.DATASETS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()