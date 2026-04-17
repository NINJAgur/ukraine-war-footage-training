"""
scraper-engine/db/models.py
SQLAlchemy ORM models for the scraper engine.
Shared with ml-engine and web-app/backend via DATABASE_SYNC_URL.
"""
import enum
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Enum, Index, Integer,
    String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ClipStatus(str, enum.Enum):
    PENDING = "PENDING"           # URL discovered, not yet downloaded
    DOWNLOADING = "DOWNLOADING"   # yt-dlp / Playwright download in progress
    DOWNLOADED = "DOWNLOADED"     # raw video on disk
    LABELED = "LABELED"           # auto-labeling complete, .txt files generated
    ANNOTATED = "ANNOTATED"       # annotated MP4 rendered, ready for public feed
    ERROR = "ERROR"               # something failed — check error_message


class ClipSource(str, enum.Enum):
    FUNKER530 = "funker530"
    GEOCONFIRMED = "geoconfirmed"
    KAGGLE = "kaggle"
    SUBMITTED = "submitted"       # user-submitted via public form


class Clip(Base):
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    url = Column(String(2000), nullable=False)
    url_hash = Column(String(64), nullable=False)   # SHA256 of canonical URL
    source = Column(
        Enum(ClipSource, name="clip_source"),
        nullable=False,
    )
    title = Column(String(500))
    description = Column(Text)
    channel = Column(String(200))                   # YouTube channel name, Funker530 author, etc.
    published_at = Column(DateTime)                 # original publish date if known

    # Processing state
    status = Column(
        Enum(ClipStatus, name="clip_status"),
        nullable=False,
        default=ClipStatus.PENDING,
    )
    error_message = Column(Text)

    # File paths (absolute paths on disk)
    file_path = Column(String(2000))                # raw downloaded video
    mp4_path = Column(String(2000))                 # annotated output video

    # Video metadata
    duration_seconds = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("url_hash", name="uq_clips_url_hash"),
        Index("ix_clips_status", "status"),
        Index("ix_clips_source", "source"),
        Index("ix_clips_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Clip id={self.id} source={self.source} status={self.status} title={self.title!r}>"
