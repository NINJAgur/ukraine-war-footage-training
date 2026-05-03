from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from db.models import ModelType, TrainingStage, TrainingStatus


class TrainingRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stage: TrainingStage
    model_type: ModelType
    status: TrainingStatus
    metrics: Optional[Any]
    weights_path: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    map50: Optional[float] = None

    @model_validator(mode='after')
    def _extract_map50(self) -> 'TrainingRunOut':
        m = self.metrics or {}
        key = next((k for k in m if 'map50' in k.lower() and 'map50-95' not in k.lower()), None)
        if key:
            try: self.map50 = round(float(m[key]), 3)
            except (ValueError, TypeError): pass
        return self


class TrainRequest(BaseModel):
    model_type: ModelType
    stage: TrainingStage = TrainingStage.BASELINE