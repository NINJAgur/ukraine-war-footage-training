"""
Unit tests for _maybe_trigger_finetune in package_dataset.
Mocks get_session and the merged-image count — no DB, GCS or GPU needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from config import settings
from db.models import ModelType
from tasks.package_dataset import _maybe_trigger_finetune

AIRCRAFT_MIN = settings.YOLO_FINETUNE_MIN_IMAGES["AIRCRAFT"]
LOWEST_MIN = min(settings.YOLO_FINETUNE_MIN_IMAGES.values())


def _make_session_ctx(active_run=None, packaged_count=0):
    """Build a mock get_session context manager with controllable query results."""
    mock_session = MagicMock()

    packaged_datasets = [MagicMock(id=i) for i in range(packaged_count)]

    def query_side_effect(model_cls):
        q = MagicMock()
        q.filter.return_value = q
        q.first.return_value = active_run
        q.all.return_value = packaged_datasets
        q.count.return_value = 0
        return q

    mock_session.query.side_effect = query_side_effect
    mock_session.get.return_value = None

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


def _run(image_counts, active_run=None, packaged_count=3):
    """Run _maybe_trigger_finetune with a fixed per-model merged image count."""
    counts = (
        image_counts if isinstance(image_counts, dict)
        else {m.value: image_counts for m in ModelType}
    )
    mock_run = MagicMock()
    mock_run.id = 99

    with patch("tasks.package_dataset.get_session",
               return_value=_make_session_ctx(active_run, packaged_count)), \
         patch("tasks.package_dataset._count_merged_images",
               side_effect=lambda mt: counts.get(mt.value, 0)), \
         patch("tasks.package_dataset._latest_weights", return_value="/w/best.pt"), \
         patch("tasks.package_dataset.TrainingRun", return_value=mock_run), \
         patch("tasks.package_dataset.celery_app") as mock_app:
        _maybe_trigger_finetune()
    return mock_app


@pytest.mark.unit
def test_does_not_dispatch_below_image_threshold():
    """Below the lowest per-model threshold, no model qualifies."""
    mock_app = _run(image_counts=LOWEST_MIN - 1)
    mock_app.send_task.assert_not_called()


@pytest.mark.unit
def test_does_not_dispatch_when_no_images_at_all():
    mock_app = _run(image_counts=0)
    mock_app.send_task.assert_not_called()


@pytest.mark.unit
def test_does_not_dispatch_when_active_run_exists():
    active = MagicMock()
    active.id = 7
    mock_app = _run(image_counts=100_000, active_run=active)
    mock_app.send_task.assert_not_called()


@pytest.mark.unit
def test_dispatches_prepare_finetune_batch_when_threshold_met():
    mock_app = _run(image_counts=100_000)

    mock_app.send_task.assert_called_once()
    assert mock_app.send_task.call_args[0][0] == "tasks.package_dataset.prepare_finetune_batch"
    assert mock_app.send_task.call_args[1]["queue"] == "pipeline"


@pytest.mark.unit
def test_dispatches_only_models_that_meet_their_own_threshold():
    """AIRCRAFT qualifies, the rest do not — one batch, one run id."""
    counts = {m.value: 0 for m in ModelType}
    counts["AIRCRAFT"] = AIRCRAFT_MIN
    mock_app = _run(image_counts=counts)

    mock_app.send_task.assert_called_once()
    assert mock_app.send_task.call_args[1]["kwargs"]["run_ids"] == [99]


@pytest.mark.unit
def test_thresholds_defined_for_every_model():
    for model_type in ModelType:
        assert settings.YOLO_FINETUNE_MIN_IMAGES[model_type.value] > 0
