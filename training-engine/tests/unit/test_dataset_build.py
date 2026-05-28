"""
Unit tests for dataset-build helpers in train_baseline.
No DB, no GPU, no network required.
"""
import pytest

from tasks.train_baseline import IMG_EXTS, _count_images


@pytest.mark.unit
def test_owner_prefix_naming_convention():
    # Kaggle dataset handles are formatted as "owner/dataset-name"
    handle = "nzigulic/ukraine-war-dataset"
    owner_prefix = handle.split("/")[0] + "_"
    assert owner_prefix == "nzigulic_"


@pytest.mark.unit
def test_owner_prefix_with_different_handle():
    handle = "mihprofi/military-vehicles-yolo"
    owner_prefix = handle.split("/")[0] + "_"
    assert owner_prefix == "mihprofi_"


@pytest.mark.unit
def test_count_images_counts_only_image_extensions(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.jpeg", "c.png", "d.bmp", "e.webp"):
        (img_dir / name).write_text("fake")
    assert _count_images(img_dir) == 5


@pytest.mark.unit
def test_count_images_excludes_non_image_files(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "label.txt").write_text("0 0.5 0.5 0.3 0.3")
    (img_dir / "data.yaml").write_text("nc: 3")
    (img_dir / "readme.md").write_text("docs")
    (img_dir / "video.mp4").write_bytes(b"\x00")
    assert _count_images(img_dir) == 0


@pytest.mark.unit
def test_count_images_mixed_files(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "frame_0001.jpg").write_text("fake")
    (img_dir / "frame_0002.PNG").write_text("fake")  # uppercase extension
    (img_dir / "annotation.txt").write_text("0 0.5 0.5 0.1 0.1")
    (img_dir / "notes.json").write_text("{}")
    # _count_images does suffix.lower() so .PNG should match
    assert _count_images(img_dir) == 2


@pytest.mark.unit
def test_count_images_empty_dir(tmp_path):
    img_dir = tmp_path / "empty"
    img_dir.mkdir()
    assert _count_images(img_dir) == 0


@pytest.mark.unit
def test_img_exts_contains_expected_formats():
    assert ".jpg" in IMG_EXTS
    assert ".jpeg" in IMG_EXTS
    assert ".png" in IMG_EXTS
    assert ".bmp" in IMG_EXTS
    assert ".webp" in IMG_EXTS
    assert ".mp4" not in IMG_EXTS
    assert ".txt" not in IMG_EXTS
