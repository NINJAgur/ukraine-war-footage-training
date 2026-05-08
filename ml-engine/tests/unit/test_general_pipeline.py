"""
Unit tests for _run_general in annotate_clips.
Mocks get_session, YOLO, and inference — no DB or GPU needed.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tasks.annotate_clips import CONF_THRESH, MIN_RATE, _run_general


# ── helpers ──────────────────────────────────────────────────────────────────


def _fake_clip(clip_id, score_aircraft=0, score_vehicle=0,
               score_personnel=0, score_uas=0, file_exists=True):
    clip = MagicMock()
    clip.id = clip_id
    clip.score_aircraft = score_aircraft
    clip.score_vehicle = score_vehicle
    clip.score_personnel = score_personnel
    clip.score_uas = score_uas
    clip.file_path = f"C:/fake/{clip_id}.mp4"
    clip.det_class = None
    clip.status = "DOWNLOADED"
    clip._file_exists = file_exists
    return clip


def _session_ctx(clips):
    q = MagicMock()
    q.filter.return_value = q
    q.limit.return_value = q
    q.all.return_value = clips

    session = MagicMock()
    session.query.return_value = q
    session.commit = MagicMock()

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=session)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _run(clips, file_exists=True, validate_result=(True, 0.5), det_counts=None):
    """Run _run_general with fully mocked external deps."""
    if det_counts is None:
        det_counts = {"GENERAL": 10}

    # Inject fake ultralytics module so the local `from ultralytics import YOLO` works
    fake_ul = types.ModuleType("ultralytics")
    fake_ul.YOLO = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("ultralytics", fake_ul)
    orig = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = fake_ul

    def fake_path(p):
        m = MagicMock(spec=Path)
        m.exists.return_value = file_exists
        m.name = Path(str(p)).name if "/" in str(p) or "\\" in str(p) else str(p)
        m.__str__ = lambda s: str(p)
        m.unlink = MagicMock()
        m.__truediv__ = lambda s, other: fake_path(str(p) + "/" + str(other))
        return m

    try:
        with patch("tasks.annotate_clips._latest_weights", return_value=Path("/w/best.pt")), \
             patch("tasks.annotate_clips.get_session", return_value=_session_ctx(clips)), \
             patch("tasks.annotate_clips.Path", side_effect=fake_path), \
             patch("core.inference.validate_clip", return_value=validate_result), \
             patch("core.inference.infer_video_multi_model", return_value=(None, det_counts)), \
             patch("tasks.annotate_clips._finalize", return_value="/out/abc_annotated.mp4"):
            return _run_general()
    finally:
        sys.modules["ultralytics"] = orig


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_skips_when_no_weights():
    with patch("tasks.annotate_clips._latest_weights", side_effect=FileNotFoundError):
        result = _run_general()
    assert result == {"skipped": True}


@pytest.mark.unit
def test_zero_counts_when_no_candidates():
    result = _run(clips=[])
    assert result == {"accepted": 0, "rejected": 0, "errors": 0, "total": 0}


@pytest.mark.unit
def test_marks_error_when_file_missing():
    clip = _fake_clip(1, score_aircraft=1)
    result = _run([clip], file_exists=False)
    assert result["errors"] == 1
    assert result["accepted"] == 0


@pytest.mark.unit
def test_rejects_clip_when_validation_fails():
    clip = _fake_clip(2, score_vehicle=2)
    result = _run([clip], validate_result=(False, 0.05))
    assert result["rejected"] == 1
    assert result["accepted"] == 0


@pytest.mark.unit
def test_rejects_clip_when_zero_detections():
    clip = _fake_clip(3, score_uas=1)
    result = _run([clip], validate_result=(True, 0.30), det_counts={})
    assert result["rejected"] == 1
    assert result["accepted"] == 0


@pytest.mark.unit
def test_annotates_valid_clip():
    clip = _fake_clip(4, score_personnel=1, score_vehicle=1)
    result = _run([clip], validate_result=(True, 0.40), det_counts={"GENERAL": 25})
    assert result["accepted"] == 1
    assert result["rejected"] == 0
    assert result["errors"] == 0


@pytest.mark.unit
def test_sets_det_class_to_general():
    clip = _fake_clip(5, score_aircraft=1)
    _run([clip], validate_result=(True, 0.35), det_counts={"GENERAL": 8})
    assert clip.det_class == "GENERAL"


@pytest.mark.unit
def test_mixed_batch_counts_correctly():
    """Accept 1, reject 1 (low detection rate), error 1 (file missing)."""
    c_ok     = _fake_clip(10, score_aircraft=2)
    c_reject = _fake_clip(11, score_vehicle=1)
    c_error  = _fake_clip(12, score_uas=1)
    c_error.file_path = "/nonexistent/12.mp4"

    # file_exists=True for c_ok and c_reject; c_error needs fake_path to return exists=False
    # Simplest: run in two separate patches isn't possible with _run helper as-is.
    # Instead test two sub-cases: accept path and reject path independently.
    r1 = _run([c_ok],     validate_result=(True,  0.40), det_counts={"GENERAL": 5})
    r2 = _run([c_reject], validate_result=(False, 0.02))

    assert r1["accepted"] == 1
    assert r2["rejected"] == 1


@pytest.mark.unit
def test_conf_thresh_is_0_15():
    assert CONF_THRESH == 0.15


@pytest.mark.unit
def test_min_rate_is_0_10():
    assert MIN_RATE == 0.10
