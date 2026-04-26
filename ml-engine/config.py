"""
ml-engine/config.py
Loads all ML-engine configuration from environment variables.
"""
from pathlib import Path
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
    ANNOTATED_VIDEO_DIR: Path = Path(__file__).parent / "media" / "annotated"
    FRAMES_DIR: Path = Path(__file__).parent / "media" / "frames"
    DATASETS_DIR: Path = Path(__file__).parent / "media" / "scraped_datasets"

    # ── Training Runs — under ml-engine/ ─────────────────────────────
    RUNS_DIR: Path = Path(__file__).parent / "runs"

    # ── Kaggle dataset cache — kept inside the project ────────────────
    KAGGLE_CACHE_DIR: Path = Path(__file__).parent / "media" / "kaggle_datasets"

    # ── GPU / Model ───────────────────────────────────────────────────
    GPU_DEVICE: str = "cuda:0"          # device for training and inference
    YOLO_MODEL: str = "yolov8m.pt"      # base model for Stage 1 baseline
    YOLO_BATCH_SIZE: int = 8            # max for 8GB VRAM with yolov8m
    YOLO_IMG_SIZE: int = 640
    YOLO_EPOCHS_BASELINE: int = 10   # increase as model quality improves
    YOLO_EPOCHS_FINETUNE: int = 10   # increase as model quality improves

    # ── Auto-labeling ─────────────────────────────────────────────────
    GDINO_CONFIG: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GDINO_CHECKPOINT: str = "groundingdino_swint_ogc.pth"
    GDINO_BOX_THRESHOLD: float = 0.35
    GDINO_TEXT_THRESHOLD: float = 0.25
    # 3-class prompt — order determines class index in .txt labels:
    # 0=aircraft, 1=vehicle, 2=personnel
    GDINO_TEXT_PROMPT: str = (
        "aircraft . drone . helicopter . missile . jet . "
        "tank . armored vehicle . military vehicle . artillery . radar . apc . "
        "soldier . fighter . personnel . combatant"
    )
    FRAME_INTERVAL: int = 30            # extract every Nth frame (30 = 1fps @ 30fps)
    MAX_FRAMES_PER_CLIP: int = 300      # cap per clip

    # ── Multi-model class definitions ─────────────────────────────────
    # Universal 3-class vocabulary — all models share the same class IDs.
    # 0=AIRCRAFT  1=VEHICLE  2=PERSONNEL
    MODEL_CLASSES: dict = {
        "GENERAL":   ["aircraft", "vehicle", "personnel"],
        "AIRCRAFT":  ["aircraft", "vehicle", "personnel"],
        "VEHICLE":   ["aircraft", "vehicle", "personnel"],
        "PERSONNEL": ["aircraft", "vehicle", "personnel"],
    }
    # Maps GDINO prompt-term index → ModelType (used by auto_label to tag datasets)
    GDINO_CLASS_TO_MODEL: dict = {
        0: "AIRCRAFT", 1: "AIRCRAFT", 2: "AIRCRAFT", 3: "AIRCRAFT", 4: "AIRCRAFT",
        5: "VEHICLE",  6: "VEHICLE",  7: "VEHICLE",  8: "VEHICLE",  9: "VEHICLE",  10: "VEHICLE",
        11: "PERSONNEL", 12: "PERSONNEL", 13: "PERSONNEL", 14: "PERSONNEL",
    }
    # Render colours per model type (BGR for OpenCV)
    MODEL_COLORS: dict = {
        "GENERAL":   (200, 200, 200),  # light grey
        "AIRCRAFT":  (255, 160,   0),  # cyan-blue
        "VEHICLE":   (0,   200,  60),  # green
        "PERSONNEL": (0,   80,  255),  # red-orange
    }

    def model_post_init(self, __context):
        for d in [
            self.ANNOTATED_VIDEO_DIR,
            self.FRAMES_DIR,
            self.DATASETS_DIR,
            self.RUNS_DIR,
            self.KAGGLE_CACHE_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
