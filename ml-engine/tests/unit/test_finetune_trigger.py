"""
Unit tests for _maybe_trigger_finetune in annotate_clips.
Mocks get_session — no DB or GPU needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from tasks.annotate_clips import FINETUNE_MIN_DATASETS, _maybe_trigger_finetune


def _make_session_ctx(active_run=None, packaged_count=0):
    """Build a mock get_session context manager with controllable query results."""
    mock_session = MagicMock()

    packaged_datasets = [MagicMock(id=i) for i in range(packaged_count)]

    # query().filter().filter().filter().first() → active_run
    # query().filter().all() → packaged_datasets
    def query_side_effect(model_cls):
        q = MagicMock()
        q.filter.return_value = q
        q.first.return_value = active_run
        q.all.return_value = packaged_datasets
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

    with patch("tasks.annotate_clips.get_session", return_value=ctx), \
         patch("tasks.annotate_clips._latest_weights", return_value=MagicMock()), \
         patch("tasks.annotate_clips.train_finetune", create=True) as mock_task:
        _maybe_trigger_finetune()

    mock_task.delay.assert_not_called()


@pytest.mark.unit
def test_does_not_dispatch_when_active_finetune_exists():
    active = MagicMock()
    active.id = 7
    ctx = _make_session_ctx(active_run=active, packaged_count=FINETUNE_MIN_DATASETS + 2)

    with patch("tasks.annotate_clips.get_session", return_value=ctx), \
         patch("tasks.annotate_clips._latest_weights", return_value=MagicMock()), \
         patch("tasks.annotate_clips.train_finetune", create=True) as mock_task:
        _maybe_trigger_finetune()

    mock_task.delay.assert_not_called()


@pytest.mark.unit
def test_dispatches_when_enough_datasets_and_no_active_run():
    ctx = _make_session_ctx(active_run=None, packaged_count=FINETUNE_MIN_DATASETS)

    mock_run_instance = MagicMock()
    mock_run_instance.id = 99

    mock_session_inner = ctx.__enter__.return_value
    mock_session_inner.add.side_effect = lambda r: None

    # train_finetune is imported locally inside _maybe_trigger_finetune, so patch it at source
    with patch("tasks.annotate_clips.get_session", return_value=ctx), \
         patch("tasks.annotate_clips._latest_weights", side_effect=FileNotFoundError("no weights")), \
         patch("tasks.annotate_clips.TrainingRun", return_value=mock_run_instance) as MockRun, \
         patch("tasks.train_finetune.train_finetune") as mock_finetune:
        _maybe_trigger_finetune()

    MockRun.assert_called_once()
    mock_finetune.delay.assert_called_once()


@pytest.mark.unit
def test_finetune_min_datasets_constant_is_5():
    assert FINETUNE_MIN_DATASETS == 5
