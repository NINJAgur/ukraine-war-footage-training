"""
ml-engine/db/models.py
SQLAlchemy ORM models shared across all services.
Clip is identical to scraper-engine/db/models.py.
Dataset and TrainingRun are ML-engine additions.
"""
import enum
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Enum, ForeignKey, Index, Integer,
    JSON, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Clip ──────────────────────────────────────────────────────────────

class ClipStatus(str, enum.Enum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    QUEUED = "QUEUED"        # dispatched to ml-engine gpu queue
    LABELED = "LABELED"
    ANNOTATED = "ANNOTATED"
    ERROR = "ERROR"


class ClipSource(str, enum.Enum):
    FUNKER530 = "funker530"
    GEOCONFIRMED = "geoconfirmed"
    KAGGLE = "kaggle"
    SUBMITTED = "submitted"


class Clip(Base):
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2000), nullable=False)
    url_hash = Column(String(64), nullable=False)
    source = Column(Enum(ClipSource, name="clip_source"), nullable=False)
    title = Column(String(500))
    description = Column(Text)
    channel = Column(String(200))
    published_at = Column(DateTime)
    status = Column(
        Enum(ClipStatus, name="clip_status"),
        nullable=False,
        default=ClipStatus.PENDING,
    )
    error_message = Column(Text)
    file_path = Column(String(2000))
    mp4_path = Column(String(2000))
    duration_seconds = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    datasets = relationship("Dataset", back_populates="clip")

    __table_args__ = (
        UniqueConstraint("url_hash", name="uq_clips_url_hash"),
        Index("ix_clips_status", "status"),
        Index("ix_clips_source", "source"),
        Index("ix_clips_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Clip id={self.id} source={self.source} status={self.status}>"


# ── Dataset ───────────────────────────────────────────────────────────

class DatasetStatus(str, enum.Enum):
    LABELED = "LABELED"     # auto-label complete, .txt files on disk
    PACKAGED = "PACKAGED"   # YOLO dir structure + data.yaml built
    TRAINED = "TRAINED"     # used in at least one TrainingRun


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    clip_id = Column(Integer, ForeignKey("clips.id"), nullable=True)
    yolo_dir_path = Column(String(2000))
    yaml_path = Column(String(2000))
    status = Column(
        Enum(DatasetStatus, name="dataset_status"),
        nullable=False,
        default=DatasetStatus.LABELED,
    )
    frame_count = Column(Integer, default=0)
    class_count = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    clip = relationship("Clip", back_populates="datasets")

    __table_args__ = (
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_clip_id", "clip_id"),
    )

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name={self.name!r} status={self.status}>"


# ── TrainingRun ───────────────────────────────────────────────────────

class ModelType(str, enum.Enum):
    GENERAL  = "GENERAL"   # all classes — generalist baseline
    SOLDIER  = "SOLDIER"   # infantry / combatants
    VEHICLE  = "VEHICLE"   # tanks, APCs, artillery, military vehicles
    AIRCRAFT = "AIRCRAFT"  # fixed-wing, helicopters, drones / UAVs


class TrainingStage(str, enum.Enum):
    BASELINE = "BASELINE"   # Stage 1: Kaggle military datasets
    FINETUNE = "FINETUNE"   # Stage 2: custom auto-labeled data


class TrainingStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stage = Column(Enum(TrainingStage, name="training_stage"), nullable=False)
    model_type = Column(Enum(ModelType, name="model_type"), nullable=False, default=ModelType.GENERAL)
    status = Column(
        Enum(TrainingStatus, name="training_status"),
        nullable=False,
        default=TrainingStatus.QUEUED,
    )
    dataset_ids = Column(JSON)          # list of Dataset.id values used
    weights_path = Column(String(2000)) # path to best.pt output
    baseline_weights = Column(String(2000))  # input baseline .pt for finetune
    metrics = Column(JSON)              # ultralytics results dict
    error_message = Column(Text)
    celery_task_id = Column(String(200))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_training_runs_status", "status"),
        Index("ix_training_runs_stage", "stage"),
        Index("ix_training_runs_model_type", "model_type"),
    )

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} stage={self.stage} model_type={self.model_type} status={self.status}>"
