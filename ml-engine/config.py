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
    DATASETS_DIR: Path = Path(__file__).parent / "media" / "datasets"

    # ── Training Runs — under ml-engine/ ─────────────────────────────
    RUNS_DIR: Path = Path(__file__).parent / "runs"

    # ── GPU / Model ───────────────────────────────────────────────────
    GPU_DEVICE: str = "cuda:0"          # device for training and inference
    YOLO_MODEL: str = "yolov8m.pt"      # base model for Stage 1 baseline
    YOLO_BATCH_SIZE: int = 8            # max for 8GB VRAM with yolov8m
    YOLO_IMG_SIZE: int = 640
    YOLO_EPOCHS_BASELINE: int = 50
    YOLO_EPOCHS_FINETUNE: int = 30

    # ── Auto-labeling ─────────────────────────────────────────────────
    GDINO_CONFIG: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GDINO_CHECKPOINT: str = "groundingdino_swint_ogc.pth"
    GDINO_BOX_THRESHOLD: float = 0.35
    GDINO_TEXT_THRESHOLD: float = 0.25
    # Combined prompt — all 8 classes, order determines class index in .txt labels:
    # 0=soldier, 1=tank, 2=armored vehicle, 3=military vehicle,
    # 4=artillery, 5=aircraft, 6=helicopter, 7=drone
    GDINO_TEXT_PROMPT: str = (
        "soldier, tank, armored vehicle, military vehicle, "
        "artillery, aircraft, helicopter, drone"
    )
    FRAME_INTERVAL: int = 30            # extract every Nth frame (30 = 1fps @ 30fps)
    MAX_FRAMES_PER_CLIP: int = 300      # cap per clip

    # ── Multi-model class definitions ─────────────────────────────────
    # Maps ModelType → class names used for YOLO training and rendering.
    # GENERAL uses all 8 GDINO classes; specialists use a subset.
    MODEL_CLASSES: dict = {
        "GENERAL":  ["soldier", "tank", "armored vehicle", "military vehicle",
                     "artillery", "aircraft", "helicopter", "drone"],
        "SOLDIER":  ["soldier"],
        "VEHICLE":  ["tank", "armored vehicle", "military vehicle", "artillery"],
        "AIRCRAFT": ["aircraft", "helicopter", "drone"],
    }
    # Maps GDINO class index → ModelType string (for label filtering in fine-tune)
    GDINO_CLASS_TO_MODEL: dict = {
        0: "SOLDIER",                          # soldier
        1: "VEHICLE", 2: "VEHICLE",            # tank, armored vehicle
        3: "VEHICLE", 4: "VEHICLE",            # military vehicle, artillery
        5: "AIRCRAFT", 6: "AIRCRAFT",          # aircraft, helicopter
        7: "AIRCRAFT",                         # drone
    }
    # Render colours per model type (BGR for OpenCV)
    MODEL_COLORS: dict = {
        "GENERAL":  (200, 200, 200),  # light grey
        "SOLDIER":  (0,   80,  255),  # red-orange
        "VEHICLE":  (0,   200,  60),  # green
        "AIRCRAFT": (255, 160,   0),  # cyan-blue
    }

    def model_post_init(self, __context):
        for d in [
            self.ANNOTATED_VIDEO_DIR,
            self.FRAMES_DIR,
            self.DATASETS_DIR,
            self.RUNS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
