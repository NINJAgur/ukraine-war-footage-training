import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from db.models import ClipSource, ClipStatus

_AIRCRAFT_RE  = re.compile(r'\b(aircraft|drone|uav|fpv|helicopter|jet|plane|shahed|geran|lancet|bayraktar|mi-28|mi-24|mi-8|su-\d+|zala|gerbera|orlan)\b', re.I)
_VEHICLE_RE   = re.compile(r'\b(tank|vehicle|apc|bmp|t-72|t-80|t-90|armor|armour|artillery|howitzer|convoy|truck|btr|bmd|grad|2s\d|tor|s-300|mlrs)\b', re.I)
_PERSONNEL_RE = re.compile(r'\b(soldier|infantry|personnel|troop|fighter|squad|soldier|troops)\b', re.I)


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
        text = f"{self.title or ''} {self.description or ''}"
        if _AIRCRAFT_RE.search(text):
            self.det_class = 'AIRCRAFT'
        elif _VEHICLE_RE.search(text):
            self.det_class = 'VEHICLE'
        elif _PERSONNEL_RE.search(text):
            self.det_class = 'PERSONNEL'
        if self.mp4_path:
            self.video_url = '/media/annotated/' + Path(self.mp4_path).name
        return self


class ClipSubmit(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
