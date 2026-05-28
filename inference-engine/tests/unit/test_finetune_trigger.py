"""
Unit tests for _maybe_trigger_finetune in package_dataset.
Mocks get_session — no DB or GPU needed.
"""
from unittest.mock import MagicMock, call, patch

import pytest

from tasks.package_dataset import FINETUNE_MIN_DATASETS, _maybe_trigger_finetune


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
    mock_session.add = MagicMock()
    mock_session.flush = MagicMock()

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


@pytest.mark.unit
def test_does_not_dispatch_when_fewer_than_min_datasets():
    ctx = _make_session_ctx(active_run=None, packaged_count=FINETUNE_MIN_DATASETS - 1)

    with patch("tasks.package_dataset.get_session", return_value=ctx), \
         patch("tasks.package_dataset._latest_weights", return_value=MagicMock()), \
         patch("tasks.package_dataset.celery_app") as mock_app:
        _maybe_trigger_finetune()

    mock_app.send_task.assert_not_called()


@pytest.mark.unit
def test_does_not_dispatch_when_active_finetune_exists():
    active = MagicMock()
    active.id = 7
    ctx = _make_session_ctx(active_run=active, packaged_count=FINETUNE_MIN_DATASETS + 2)

    with patch("tasks.package_dataset.get_session", return_value=ctx), \
         patch("tasks.package_dataset._latest_weights", return_value=MagicMock()), \
         patch("tasks.package_dataset.celery_app") as mock_app:
        _maybe_trigger_finetune()

    mock_app.send_task.assert_not_called()


@pytest.mark.unit
def test_dispatches_prepare_finetune_batch_when_enough_datasets():
    """_maybe_trigger_finetune dispatches prepare_finetune_batch (not train_finetune)."""
    ctx = _make_session_ctx(active_run=None, packaged_count=FINETUNE_MIN_DATASETS)

    mock_run = MagicMock()
    mock_run.id = 99

    with patch("tasks.package_dataset.get_session", return_value=ctx), \
         patch("tasks.package_dataset._latest_weights", side_effect=FileNotFoundError), \
         patch("tasks.package_dataset.TrainingRun", return_value=mock_run), \
         patch("tasks.package_dataset.celery_app") as mock_app:
        _maybe_trigger_finetune()

    # send_task called once with prepare_finetune_batch (run_ids may contain 1-4 entries)
    mock_app.send_task.assert_called_once()
    task_name = mock_app.send_task.call_args[0][0]
    assert task_name == "tasks.package_dataset.prepare_finetune_batch"
    queued_to = mock_app.send_task.call_args[1]["queue"]
    assert queued_to == "pipeline"


@pytest.mark.unit
def test_finetune_min_datasets_constant_is_5():
    assert FINETUNE_MIN_DATASETS == 5
