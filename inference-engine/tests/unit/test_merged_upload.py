"""
Unit tests for incremental merged-dir accumulation and upload.

Regression cover for the failure that stalled production: package_dataset used to
re-upload the whole accumulated merged tree once per clip per model, so nightly
upload cost grew quadratically until it exceeded the VM's time budget.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from db.models import ModelType
from tasks.package_dataset import _append_to_merged, _upload_merged_to_gcs


def _make_clip_dataset(root, stems):
    """Build a minimal per-clip YOLO dir with one AIRCRAFT label per stem."""
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (root / "train" / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (root / "train" / "images" / f"{stem}.jpg").write_bytes(b"jpg")
    return root


@pytest.mark.unit
def test_returns_only_files_written_this_call(tmp_path):
    merged_root = tmp_path / "datasets"
    clip_a = _make_clip_dataset(tmp_path / "clip_a", ["f1", "f2"])
    clip_b = _make_clip_dataset(tmp_path / "clip_b", ["f3"])

    with patch("tasks.package_dataset.settings") as s:
        s.DATASETS_DIR = merged_root
        s.MODEL_CLASSES = {"AIRCRAFT": ["aircraft", "vehicle", "personnel"]}

        first = _append_to_merged(clip_a, 1, ModelType.AIRCRAFT)
        second = _append_to_merged(clip_b, 2, ModelType.AIRCRAFT)

    # Second call must not re-report the first clip's files, even though they are
    # still sitting in the merged dir.
    assert not any(f.name.startswith("1_") for f in second)
    assert {f.name for f in second if f.suffix == ".jpg"} == {"2_f3.jpg"}
    assert len([f for f in first if f.suffix == ".jpg"]) == 2

    # data.yaml is rewritten every call and must ship with each batch
    assert any(f.name == "data.yaml" for f in second)


@pytest.mark.unit
def test_upload_sends_only_the_listed_files(tmp_path):
    merged_dir = tmp_path / "AIRCRAFT"
    (merged_dir / "train" / "images").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (merged_dir / "train" / "images" / name).write_bytes(b"jpg")

    blob = MagicMock()
    bucket_obj = MagicMock()
    bucket_obj.blob.return_value = blob
    fake_gcs = types.ModuleType("google.cloud.storage")
    fake_gcs.Client = MagicMock(return_value=MagicMock(bucket=MagicMock(return_value=bucket_obj)))

    only = [merged_dir / "train" / "images" / "b.jpg"]
    with patch.dict(sys.modules, {"google.cloud.storage": fake_gcs}):
        _upload_merged_to_gcs(merged_dir, "AIRCRAFT", "test-bucket", files=only)

    bucket_obj.blob.assert_called_once_with("merged/AIRCRAFT/train/images/b.jpg")


@pytest.mark.unit
def test_upload_without_files_sends_whole_tree(tmp_path):
    """prepare_finetune_batch still needs the full-tree upload."""
    merged_dir = tmp_path / "AIRCRAFT"
    (merged_dir / "train" / "images").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (merged_dir / "train" / "images" / name).write_bytes(b"jpg")

    bucket_obj = MagicMock()
    fake_gcs = types.ModuleType("google.cloud.storage")
    fake_gcs.Client = MagicMock(return_value=MagicMock(bucket=MagicMock(return_value=bucket_obj)))

    with patch.dict(sys.modules, {"google.cloud.storage": fake_gcs}):
        _upload_merged_to_gcs(merged_dir, "AIRCRAFT", "test-bucket")

    assert bucket_obj.blob.call_count == 3
