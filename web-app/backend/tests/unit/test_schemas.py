"""
Unit tests for TrainingRunOut schema — specifically the map50 model_validator.
No DB or network needed.
"""
from datetime import datetime

import pytest

from schemas.training import TrainingRunOut
from db.models import ModelType, TrainingStage, TrainingStatus


def _base_data(**kwargs):
    defaults = {
        "id": 1,
        "stage": TrainingStage.BASELINE,
        "model_type": ModelType.AIRCRAFT,
        "status": TrainingStatus.DONE,
        "metrics": None,
        "weights_path": None,
        "started_at": None,
        "completed_at": None,
        "created_at": datetime(2025, 1, 1),
    }
    defaults.update(kwargs)
    return defaults


@pytest.mark.unit
def test_map50_extracted_from_metrics():
    data = _base_data(metrics={"metrics/mAP50(B)": 0.929, "metrics/mAP50-95(B)": 0.6})
    out = TrainingRunOut(**data)
    assert out.map50 == 0.929


@pytest.mark.unit
def test_map50_rounds_to_3_decimal_places():
    data = _base_data(metrics={"metrics/mAP50(B)": 0.87149999})
    out = TrainingRunOut(**data)
    assert out.map50 == round(0.87149999, 3)


@pytest.mark.unit
def test_map50_none_when_metrics_is_none():
    data = _base_data(metrics=None)
    out = TrainingRunOut(**data)
    assert out.map50 is None


@pytest.mark.unit
def test_map50_none_when_only_map50_95_key_exists():
    data = _base_data(metrics={"metrics/mAP50-95(B)": 0.6})
    out = TrainingRunOut(**data)
    assert out.map50 is None


@pytest.mark.unit
def test_map50_none_when_metrics_empty_dict():
    data = _base_data(metrics={})
    out = TrainingRunOut(**data)
    assert out.map50 is None


@pytest.mark.unit
def test_map50_none_when_value_not_numeric():
    data = _base_data(metrics={"metrics/mAP50(B)": "not-a-number"})
    out = TrainingRunOut(**data)
    assert out.map50 is None


@pytest.mark.unit
def test_model_type_and_stage_preserved():
    data = _base_data(
        metrics={"metrics/mAP50(B)": 0.780},
        model_type=ModelType.PERSONNEL,
        stage=TrainingStage.FINETUNE,
    )
    out = TrainingRunOut(**data)
    assert out.model_type == ModelType.PERSONNEL
    assert out.stage == TrainingStage.FINETUNE
    assert out.map50 == 0.780
