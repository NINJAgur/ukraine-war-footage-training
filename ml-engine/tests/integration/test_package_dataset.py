"""
Integration tests for create_train_val_split and update_data_yaml.
Uses tmp_path only — no DB, no GPU needed.

Run with:
    pytest -m integration tests/integration/test_package_dataset.py
"""
import yaml
import pytest

from tasks.package_dataset import VAL_SPLIT, create_train_val_split, update_data_yaml


def _make_dataset(tmp_path, n_images=10):
    """Create a minimal YOLO dataset folder with n_images in train/images."""
    train_img = tmp_path / "train" / "images"
    train_lbl = tmp_path / "train" / "labels"
    train_img.mkdir(parents=True)
    train_lbl.mkdir(parents=True)
    for i in range(n_images):
        (train_img / f"frame_{i:04d}.jpg").write_text("fake")
        (train_lbl / f"frame_{i:04d}.txt").write_text("0 0.5 0.5 0.3 0.3")
    return tmp_path


@pytest.mark.integration
def test_val_split_moves_roughly_20_percent(tmp_path):
    n = 20
    _make_dataset(tmp_path, n_images=n)
    train_count, val_count = create_train_val_split(tmp_path)
    assert val_count == max(1, int(n * VAL_SPLIT))
    assert train_count == n - val_count


@pytest.mark.integration
def test_val_images_dir_created(tmp_path):
    _make_dataset(tmp_path, n_images=10)
    create_train_val_split(tmp_path)
    assert (tmp_path / "val" / "images").exists()
    assert (tmp_path / "val" / "labels").exists()


@pytest.mark.integration
def test_val_images_count_matches_returned_count(tmp_path):
    _make_dataset(tmp_path, n_images=10)
    _, val_count = create_train_val_split(tmp_path)
    actual_val = list((tmp_path / "val" / "images").glob("*.jpg"))
    assert len(actual_val) == val_count


@pytest.mark.integration
def test_train_images_count_matches_returned_count(tmp_path):
    _make_dataset(tmp_path, n_images=10)
    train_count, _ = create_train_val_split(tmp_path)
    actual_train = list((tmp_path / "train" / "images").glob("*.jpg"))
    assert len(actual_train) == train_count


@pytest.mark.integration
def test_val_labels_moved_alongside_images(tmp_path):
    _make_dataset(tmp_path, n_images=10)
    _, val_count = create_train_val_split(tmp_path)
    val_labels = list((tmp_path / "val" / "labels").glob("*.txt"))
    assert len(val_labels) == val_count


@pytest.mark.integration
def test_total_images_unchanged_after_split(tmp_path):
    n = 15
    _make_dataset(tmp_path, n_images=n)
    train_count, val_count = create_train_val_split(tmp_path)
    assert train_count + val_count == n


@pytest.mark.integration
def test_minimum_one_val_image(tmp_path):
    # Even with 1 image, val_count must be at least 1
    _make_dataset(tmp_path, n_images=1)
    train_count, val_count = create_train_val_split(tmp_path)
    assert val_count >= 1


@pytest.mark.integration
def test_update_data_yaml_writes_correct_keys(tmp_path):
    _make_dataset(tmp_path, n_images=5)
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("nc: 3\nnames: [AIRCRAFT, VEHICLE, PERSONNEL]\n")

    update_data_yaml(yaml_path, tmp_path)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    assert data["path"] == str(tmp_path)
    assert data["train"] == "train/images"
    assert data["val"] == "val/images"


@pytest.mark.integration
def test_update_data_yaml_preserves_nc_and_names(tmp_path):
    _make_dataset(tmp_path, n_images=5)
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("nc: 3\nnames: [AIRCRAFT, VEHICLE, PERSONNEL]\n")

    update_data_yaml(yaml_path, tmp_path)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    assert data["nc"] == 3
    assert data["names"] == ["AIRCRAFT", "VEHICLE", "PERSONNEL"]
