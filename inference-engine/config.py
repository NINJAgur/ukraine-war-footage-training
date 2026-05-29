"""
inference-engine/config.py
Loads all inference-engine configuration from environment variables.
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
    ANNOTATED_VIDEO_DIR: Path = Path(__file__).parent / "media"
    DATASETS_DIR: Path = Path(__file__).parent / "media" / "scraped_datasets"
    FRAMES_DIR: Path = Path(__file__).parent / "media" / "scraped_datasets" / "frames"

    # ── Storage Mode (local or remote) ────────────────────────────────
    STORAGE_MODE: str = "local"
    REMOTE_STORAGE_BUCKET: str = "ukraine-footage-bucket"

    # ── GPU / Model ───────────────────────────────────────────────────
    GPU_DEVICE: str = "cuda:0"          # device for inference

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
    # Render colours per model type (BGR for OpenCV) — matched to CSS oklch vars in style.css
    MODEL_COLORS: dict = {
        "GENERAL":   (  0, 105, 223),  # BGR ← oklch(0.65 0.18 55)  rgb(223,105,0)
        "AIRCRAFT":  (200, 153,   0),  # BGR ← oklch(0.62 0.16 220) rgb(0,153,200)
        "VEHICLE":   ( 61,  59, 222),  # BGR ← oklch(0.60 0.20 25)  rgb(222,59,61)
        "PERSONNEL": ( 48, 154,  24),  # BGR ← oklch(0.60 0.18 145) rgb(24,154,48)
    }

    # ── Fine-tune trigger threshold ───────────────────────────────────
    YOLO_FINETUNE_MAX_CYCLES: int = 4   # baseline(10) + 4×finetune(10) = 50 total epochs

    def model_post_init(self, __context):
        # Resolve GDINO config via installed package — the default relative path
        # "groundingdino/config/..." doesn't exist inside Docker containers.
        cfg = Path(self.GDINO_CONFIG)
        if not cfg.is_absolute() or not cfg.exists():
            try:
                import groundingdino
                self.GDINO_CONFIG = str(
                    Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
                )
            except ImportError:
                pass  # not in inference-engine context; GDINO_CONFIG unused here
        for d in [
            self.ANNOTATED_VIDEO_DIR,
            self.FRAMES_DIR,
            self.DATASETS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()