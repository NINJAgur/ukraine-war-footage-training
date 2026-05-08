"""
Smoke tests: model weights exist, DB reachable, GPU available.

Run with:
    pytest -m smoke tests/smoke/test_smoke.py
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.mark.smoke
@pytest.mark.gpu
def test_aircraft_best_pt_exists():
    runs_dir = _ROOT / "runs" / "baseline" / "AIRCRAFT"
    weights = list(runs_dir.glob("*/weights/best.pt")) if runs_dir.exists() else []
    assert len(weights) > 0, f"No AIRCRAFT best.pt found under {runs_dir}"


@pytest.mark.smoke
@pytest.mark.gpu
def test_vehicle_best_pt_exists():
    runs_dir = _ROOT / "runs" / "baseline" / "VEHICLE"
    weights = list(runs_dir.glob("*/weights/best.pt")) if runs_dir.exists() else []
    assert len(weights) > 0, f"No VEHICLE best.pt found under {runs_dir}"


@pytest.mark.smoke
@pytest.mark.gpu
def test_personnel_best_pt_exists():
    runs_dir = _ROOT / "runs" / "baseline" / "PERSONNEL"
    weights = list(runs_dir.glob("*/weights/best.pt")) if runs_dir.exists() else []
    assert len(weights) > 0, f"No PERSONNEL best.pt found under {runs_dir}"


@pytest.mark.smoke
def test_db_reachable():
    from db.session import get_session
    from db.models import TrainingRun

    with get_session() as session:
        count = session.query(TrainingRun).count()
    assert isinstance(count, int)


@pytest.mark.smoke
@pytest.mark.gpu
def test_gpu_available():
    import torch
    assert torch.cuda.is_available(), "CUDA GPU not available — check drivers and CUDA installation"
