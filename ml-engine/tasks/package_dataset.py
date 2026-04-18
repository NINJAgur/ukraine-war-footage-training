"""
ml-engine/tasks/package_dataset.py

Celery task: take a LABELED Dataset, split frames into train/val (80/20),
update data.yaml with correct split paths, mark Dataset as PACKAGED,
then dispatch render_annotated for the parent Clip.

Pipeline:
  Dataset(LABELED) → train/val split → data.yaml updated → Dataset(PACKAGED)
                                                                    ↓
                                                       dispatch render_annotated
"""
import logging
import random
import shutil
from pathlib import Path

import yaml

from celery_app import celery_app
from config import settings
from db.models import Clip, Dataset, DatasetStatus
from db.session import get_session

logger = logging.getLogger(__name__)

VAL_SPLIT = 0.2
SPLIT_SEED = 42


def create_train_val_split(dataset_dir: Path) -> tuple[int, int]:
    """
    Split train/images + train/labels 80/20 into val/.
    Returns (train_count, val_count).
    Idempotent: re-calculates if val/ already exists.
    """
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    val_img_dir = dataset_dir / "val" / "images"
    val_lbl_dir = dataset_dir / "val" / "labels"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(train_img_dir.glob("*.jpg"))
    random.seed(SPLIT_SEED)
    random.shuffle(images)

    val_count = max(1, int(len(images) * VAL_SPLIT))
    val_images = images[:val_count]
    train_images = images[val_count:]

    # Move val images + labels
    for img_path in val_images:
        label_path = train_lbl_dir / (img_path.stem + ".txt")
        shutil.move(str(img_path), val_img_dir / img_path.name)
        if label_path.exists():
            shutil.move(str(label_path), val_lbl_dir / label_path.name)

    return len(train_images), val_count


def update_data_yaml(yaml_path: Path, dataset_dir: Path) -> None:
    """Rewrite data.yaml with absolute paths and correct train/val split dirs."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    data["path"] = str(dataset_dir)
    data["train"] = "train/images"
    data["val"] = "val/images"

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


@celery_app.task(
    bind=True,
    name="tasks.package_dataset.package_dataset",
    queue="gpu",
    autoretry_for=(Exception,),
    max_retries=2,
    default_retry_delay=60,
)
def package_dataset(self, dataset_id: int) -> dict:
    """
    Split a LABELED Dataset into train/val, update data.yaml, mark PACKAGED.
    Dispatches render_annotated for the parent Clip on completion.
    """
    logger.info(f"[{self.request.id}] package_dataset dataset_id={dataset_id}")

    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        if dataset.status == DatasetStatus.PACKAGED:
            logger.info(f"[{self.request.id}] Dataset {dataset_id} already packaged — skipping")
            return {"status": "skipped", "dataset_id": dataset_id}
        dataset_dir = Path(dataset.yolo_dir_path)
        yaml_path = Path(dataset.yaml_path)
        clip_id = dataset.clip_id

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    train_count, val_count = create_train_val_split(dataset_dir)
    update_data_yaml(yaml_path, dataset_dir)

    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        dataset.status = DatasetStatus.PACKAGED

    logger.info(
        f"[{self.request.id}] Dataset {dataset_id} packaged — "
        f"train={train_count}  val={val_count}"
    )

    if clip_id is not None:
        from tasks.render_annotated import render_annotated_clip
        render_annotated_clip.delay(clip_id=clip_id)

    return {
        "status": "packaged",
        "dataset_id": dataset_id,
        "train_count": train_count,
        "val_count": val_count,
    }
