from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from db.models import ClipSource, ClipStatus


class ClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    url_hash: str
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

    det_class: Optional[str] = None
    video_url: Optional[str] = None

    @model_validator(mode='after')
    def _compute_derived(self) -> 'ClipOut':
        if self.mp4_path:
            mp4 = Path(self.mp4_path)
            parts = mp4.parts
            try:
                idx = list(parts).index('annotated')
                self.video_url = '/media/annotated/' + '/'.join(parts[idx + 1:])
            except ValueError:
                self.video_url = '/media/annotated/' + mp4.name
        return self


class ClipSubmit(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
