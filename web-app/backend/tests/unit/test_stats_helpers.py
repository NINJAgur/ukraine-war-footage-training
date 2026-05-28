"""
Unit tests for _live_map50 from api/public.py.
Uses tmp_path to create fake results.csv — no DB, no network needed.
"""
import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from api.public import _live_map50


def _write_results_csv(path: Path, rows: list[dict]):
    fieldnames = ["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.unit
def test_live_map50_returns_last_row_value(tmp_path):
    model_dir = tmp_path / "AIRCRAFT" / "run_1"
    model_dir.mkdir(parents=True)
    _write_results_csv(model_dir / "results.csv", [
        {"epoch": "1", "metrics/mAP50(B)": "0.500", "metrics/mAP50-95(B)": "0.300"},
        {"epoch": "2", "metrics/mAP50(B)": "0.871", "metrics/mAP50-95(B)": "0.550"},
    ])
    with patch("api.public._RUNS_DIR", tmp_path):
        result = _live_map50("AIRCRAFT")
    assert result == 0.871


@pytest.mark.unit
def test_live_map50_returns_none_for_missing_dir(tmp_path):
    with patch("api.public._RUNS_DIR", tmp_path):
        result = _live_map50("NONEXISTENT_MODEL")
    assert result is None


@pytest.mark.unit
def test_live_map50_returns_none_for_empty_csv(tmp_path):
    model_dir = tmp_path / "VEHICLE" / "run_1"
    model_dir.mkdir(parents=True)
    (model_dir / "results.csv").write_text("epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n")
    with patch("api.public._RUNS_DIR", tmp_path):
        result = _live_map50("VEHICLE")
    assert result is None


@pytest.mark.unit
def test_live_map50_returns_none_when_no_csv(tmp_path):
    model_dir = tmp_path / "PERSONNEL" / "run_1"
    model_dir.mkdir(parents=True)
    # no results.csv
    with patch("api.public._RUNS_DIR", tmp_path):
        result = _live_map50("PERSONNEL")
    assert result is None


@pytest.mark.unit
def test_live_map50_returns_none_when_no_runs_subdir(tmp_path):
    # model dir exists but no subdirectories
    (tmp_path / "GENERAL").mkdir()
    with patch("api.public._RUNS_DIR", tmp_path):
        result = _live_map50("GENERAL")
    assert result is None


@pytest.mark.unit
def test_live_map50_rounds_to_3_decimal_places(tmp_path):
    model_dir = tmp_path / "AIRCRAFT" / "run_1"
    model_dir.mkdir(parents=True)
    _write_results_csv(model_dir / "results.csv", [
        {"epoch": "1", "metrics/mAP50(B)": "0.92901", "metrics/mAP50-95(B)": "0.6"},
    ])
    with patch("api.public._RUNS_DIR", tmp_path):
        result = _live_map50("AIRCRAFT")
    assert result == round(0.92901, 3)
