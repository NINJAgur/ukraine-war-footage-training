"""
Unit tests for _make_epoch_callback in train_baseline.
Mocks get_session so no DB is needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from tasks.train_baseline import _make_epoch_callback


def _make_trainer(epoch=2, metrics=None, loss_items=None):
    trainer = MagicMock()
    trainer.epoch = epoch
    trainer.metrics = metrics or {"metrics/mAP50(B)": 0.75, "metrics/mAP50-95(B)": 0.4}
    trainer.loss_items = loss_items
    return trainer


@pytest.mark.unit
def test_callback_writes_epoch_progress_keys():
    mock_run = MagicMock()
    mock_run.metrics = {}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_run
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=42, total_epochs=10)
        trainer = _make_trainer(epoch=2)
        callback(trainer)

    assert mock_run.metrics is not None
    progress = mock_run.metrics["epoch_progress"]
    assert "epoch" in progress
    assert "epochs" in progress
    assert "map50" in progress
    assert "box_loss" in progress
    assert "cls_loss" in progress


@pytest.mark.unit
def test_callback_epoch_value_is_one_indexed():
    mock_run = MagicMock()
    mock_run.metrics = {}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_run
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=1, total_epochs=5)
        trainer = _make_trainer(epoch=0)  # trainer.epoch is 0-indexed
        callback(trainer)

    assert mock_run.metrics["epoch_progress"]["epoch"] == 1


@pytest.mark.unit
def test_callback_total_epochs_stored():
    mock_run = MagicMock()
    mock_run.metrics = {}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_run
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=1, total_epochs=20)
        callback(_make_trainer())

    assert mock_run.metrics["epoch_progress"]["epochs"] == 20


@pytest.mark.unit
def test_callback_map50_extracted_from_trainer_metrics():
    mock_run = MagicMock()
    mock_run.metrics = {}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_run
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=1, total_epochs=10)
        trainer = _make_trainer(metrics={"metrics/mAP50(B)": 0.929})
        callback(trainer)

    assert mock_run.metrics["epoch_progress"]["map50"] == 0.929


@pytest.mark.unit
def test_callback_loss_items_stored_when_present():
    mock_run = MagicMock()
    mock_run.metrics = {}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_run
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=1, total_epochs=10)
        trainer = _make_trainer(loss_items=[0.35, 0.21, 0.15])
        callback(trainer)

    progress = mock_run.metrics["epoch_progress"]
    assert progress["box_loss"] == round(0.35, 4)
    assert progress["cls_loss"] == round(0.21, 4)


@pytest.mark.unit
def test_callback_loss_items_none_when_missing():
    mock_run = MagicMock()
    mock_run.metrics = {}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_run
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=1, total_epochs=10)
        trainer = _make_trainer(loss_items=None)
        callback(trainer)

    progress = mock_run.metrics["epoch_progress"]
    assert progress["box_loss"] is None
    assert progress["cls_loss"] is None


@pytest.mark.unit
def test_callback_preserves_existing_metrics():
    mock_run = MagicMock()
    mock_run.metrics = {"total_train_images": 8433}

    mock_session = MagicMock()
    mock_session.get.return_value = mock_run
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=1, total_epochs=10)
        callback(_make_trainer())

    assert mock_run.metrics["total_train_images"] == 8433
    assert "epoch_progress" in mock_run.metrics


@pytest.mark.unit
def test_callback_does_not_raise_when_run_not_found():
    mock_session = MagicMock()
    mock_session.get.return_value = None  # run not found
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("tasks.train_baseline.get_session", return_value=mock_ctx):
        callback = _make_epoch_callback(run_id=999, total_epochs=10)
        # Should not raise — callback swallows exceptions
        callback(_make_trainer())
