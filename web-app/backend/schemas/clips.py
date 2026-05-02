from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from db.models import ClipSource, ClipStatus


class ClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    title: Optional[str]
    description: Optional[str]
    source: ClipSource
    status: ClipStatus
    mp4_path: Optional[str]
    duration_seconds: Optional[int]
    width: Optional[int]
    height: Optional[int]
    published_at: Optional[datetime]
    created_at: datetime


class ClipSubmit(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None