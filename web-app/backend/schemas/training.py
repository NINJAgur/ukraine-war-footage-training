from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

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


class TrainRequest(BaseModel):
    model_type: ModelType
    stage: TrainingStage = TrainingStage.BASELINE