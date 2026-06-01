"""
training-engine/tasks/train_finetune.py

Celery task: Stage 2 fine-tuning — use pre-built merged dataset to train YOLOv8m.

  scraped_merged_path is passed by prepare_finetune_batch (inference-engine):
    gs://bucket/merged/<MODEL>   → download to local temp dir, then train
    /local/path/to/merged/<MODEL> → read directly (local mode, same machine)
    None                          → Kaggle merged only (no scraped data)

After training:
  remote mode: weights uploaded to gs://bucket/runs/finetune/<MODEL>/<run_name>/best.pt
  local mode:  weights stay in training-engine/runs/finetune/ (default RUNS_DIR)
  finally:     local merged dir deleted in both modes
"""
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from celery_app import celery_app
from config import settings
from db.models import ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session
from tasks.train_baseline import _make_epoch_callbacks

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def _download_merged_from_gcs(gcs_path: str, local_dir: Path) -> None:
    """Download gs://bucket/merged/<MODEL> to local_dir."""
    from google.cloud import storage as gcs
    without_scheme = gcs_path[len("gs://"):]
    bucket_name, _, prefix_base = without_scheme.partition("/")
    prefix = prefix_base.rstrip("/") + "/"
    client = gcs.Client()
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        rel_path = blob.name[len(prefix):]
        if not rel_path:
            continue
        local = local_dir / rel_path
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
    logger.info(f"[train_finetune] Downloaded {gcs_path} → {local_dir.name}")


def _upload_weights_to_gcs(local: Path, bucket: str, model: str, run_id: int) -> None:
    """Upload best.pt weights to GCS."""
    from google.cloud import storage as gcs
    blob_name = f"runs/finetune/{model}/finetune_{model}_{run_id}/weights/best.pt"
    gcs.Client().bucket(bucket).blob(blob_name).upload_from_filename(str(local))
    logger.info(f"[train_finetune] Uploaded weights → gs://{bucket}/{blob_name}")


def _delete_gcs_merged(bucket: str, model: str) -> None:
    """Delete gs://bucket/merged/MODEL/ after training — datasets are consumed, don't accumulate."""
    from google.cloud import storage as gcs
    client = gcs.Client()
    blobs = list(client.list_blobs(bucket, prefix=f"merged/{model}/"))
    if blobs:
        client.bucket(bucket).delete_blobs(blobs)
        logger.info(f"[train_finetune] Deleted gs://{bucket}/merged/{model}/ ({len(blobs)} blobs)")


def _extract_metrics(results) -> dict:
    try:
        return dict(results.results_dict)
    except Exception:
        return {}


@celery_app.task(
    bind=True,
    name="tasks.train_finetune.train_finetune",
    queue="training",
    autoretry_for=(Exception,),
    max_retries=1,
    default_retry_delay=300,
)
def train_finetune(self, training_run_id: int, scraped_merged_path: str = None) -> dict:
    """
    Stage 2: fine-tune from pre-built merged dataset.
    scraped_merged_path: gs://... (download from GCS) | local path | None (Kaggle only)
    """
    logger.info(
        f"[train_finetune] task_id={self.request.id}  training_run_id={training_run_id}  "
        f"scraped_merged_path={scraped_merged_path}"
    )

    with get_session() as session:
        run = session.get(TrainingRun, training_run_id)
        if run is None:
            raise ValueError(f"TrainingRun {training_run_id} not found")
        if run.status == TrainingStatus.DONE:
            return {"status": "skipped", "training_run_id": training_run_id}

        model_type = run.model_type
        dataset_ids = run.dataset_ids or []
        baseline_weights = run.baseline_weights or settings.YOLO_MODEL

        run.status = TrainingStatus.RUNNING
        run.celery_task_id = self.request.id
        run.started_at = datetime.utcnow()
        logger.info(
            f"[train_finetune] model={model_type.value}  datasets={len(dataset_ids)}  "
            f"epochs={settings.YOLO_EPOCHS_FINETUNE}\n"
            f"    baseline_weights={baseline_weights}"
        )

    from core.main import train_model

    run_dir = settings.RUNS_DIR / "finetune" / model_type.value
    run_name = f"finetune_{model_type.value}_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"

    # Resolve merged_dir from scraped_merged_path
    # GCS path → download to local temp; local path → use directly; None → Kaggle only
    if scraped_merged_path and scraped_merged_path.startswith("gs://"):
        merged_dir = settings.DATASETS_DIR / f"merged_{model_type.value}_{training_run_id}"
        _download_merged_from_gcs(scraped_merged_path, merged_dir)
    elif scraped_merged_path:
        merged_dir = Path(scraped_merged_path)
    else:
        merged_dir = None

    try:
        import os
        os.chdir(Path(__file__).parent.parent)  # CWD = training-engine/ so YOLO writes runs/ there

        kaggle_dir = settings.KAGGLE_CACHE_DIR / "merged" / model_type.value

        if not merged_dir or not merged_dir.exists():
            # No scraped data — fine-tune on Kaggle merged only
            yaml_path = kaggle_dir / "dataset.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"Pre-built merged dataset not found: {yaml_path}")
            total_train = len(list((kaggle_dir / "train" / "images").glob("*.jpg")))
            total_val   = len(list((kaggle_dir / "val"   / "images").glob("*.jpg")))
            logger.info(
                f"[train_finetune] {model_type.value}: Kaggle only  train={total_train}  val={total_val}"
            )
        else:
            class_names = settings.MODEL_CLASSES[model_type.value]
            yaml_path = merged_dir / "combined_data.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(
                    {
                        "train": [
                            str(kaggle_dir / "train" / "images"),
                            str(merged_dir / "train" / "images"),
                        ],
                        "val": [
                            str(kaggle_dir / "val" / "images"),
                            str(merged_dir / "val" / "images"),
                        ],
                        "nc": len(class_names),
                        "names": class_names,
                    },
                    f,
                    default_flow_style=False,
                )
            total_train = (
                len(list((kaggle_dir / "train" / "images").glob("*.jpg")))
                + len(list((merged_dir / "train" / "images").glob("*.jpg")))
            )
            total_val = (
                len(list((kaggle_dir / "val" / "images").glob("*.jpg")))
                + len(list((merged_dir / "val" / "images").glob("*.jpg")))
            )
            logger.info(
                f"[train_finetune] {model_type.value}: Kaggle + {len(dataset_ids)} scraped datasets  "
                f"train={total_train}  val={total_val}"
            )

        results = train_model(
            yaml_path=str(yaml_path),
            epochs=settings.YOLO_EPOCHS_FINETUNE,
            imgsz=settings.YOLO_IMG_SIZE,
            batch=settings.YOLO_BATCH_SIZE,
            device=settings.GPU_DEVICE,
            project=str(run_dir),
            name=run_name,
            weights=baseline_weights,
            resume=False,
            extra_callbacks=dict(zip(
                ("on_train_epoch_start", "on_fit_epoch_end"),
                _make_epoch_callbacks(training_run_id, settings.YOLO_EPOCHS_FINETUNE),
            )),
        )

        if not weights_path.exists():
            raise FileNotFoundError(f"Training finished but best.pt not found: {weights_path}")

        metrics = _extract_metrics(results)
        metrics["total_train_images"] = total_train

        if settings.STORAGE_MODE == "remote" and settings.REMOTE_STORAGE_BUCKET:
            _upload_weights_to_gcs(
                weights_path, settings.REMOTE_STORAGE_BUCKET,
                model_type.value, training_run_id,
            )
            _delete_gcs_merged(settings.REMOTE_STORAGE_BUCKET, model_type.value)

        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            run.status = TrainingStatus.DONE
            run.weights_path = str(weights_path)
            run.metrics = metrics
            run.completed_at = datetime.utcnow()
            session.flush()

        map50 = metrics.get("metrics/mAP50(B)", metrics.get("mAP50(B)", 0.0))
        logger.info(
            f"[train_finetune] {model_type.value}: -> DONE  run_id={training_run_id}  "
            f"mAP50={map50:.3f}  datasets_used={len(dataset_ids)}\n"
            f"    weights={weights_path}"
        )
        import subprocess
        if sys.platform != "win32":
            with get_session() as _s:
                remaining = _s.query(TrainingRun).filter(
                    TrainingRun.status.in_([TrainingStatus.QUEUED, TrainingStatus.RUNNING]),
                    TrainingRun.id != training_run_id,
                ).count()
            if remaining == 0:
                logger.info("[shutdown] No more QUEUED/RUNNING runs — shutting down VM")
                subprocess.run(["sudo", "shutdown", "-h", "now"])
            else:
                logger.info(f"[shutdown] {remaining} run(s) still pending — keeping VM alive")
        return {
            "status": "done",
            "training_run_id": training_run_id,
            "model_type": model_type.value,
            "weights_path": str(weights_path),
            "metrics": metrics,
            "datasets_used": len(dataset_ids),
        }

    except Exception as exc:
        logger.error(f"[train_finetune] {model_type.value}: -> ERROR  run_id={training_run_id}  {exc}")
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            if run:
                run.status = TrainingStatus.ERROR
                run.error_message = str(exc)[:2000]
                run.completed_at = datetime.utcnow()
        raise

    finally:
        if merged_dir and merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
            logger.info(f"[train_finetune] Deleted merged dir: {merged_dir.name}")
        yolo_run_dir = run_dir / run_name
        if yolo_run_dir.exists() and not weights_path.exists():
            shutil.rmtree(yolo_run_dir, ignore_errors=True)
            logger.info(f"[train_finetune] Deleted incomplete run dir: {run_name}")


if __name__ == "__main__":
    import os
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("model_type", choices=[m.value for m in ModelType])
    _parser.add_argument("--epochs", type=int, default=None)
    _args = _parser.parse_args()
    _model_type = ModelType(_args.model_type.upper())
    if _args.epochs:
        settings.YOLO_EPOCHS_FINETUNE = _args.epochs

    def _latest_weights_for(model: ModelType):
        for stage in ("finetune", "baseline"):
            runs_dir = settings.RUNS_DIR / stage / model.value
            if not runs_dir.exists():
                continue
            candidates = sorted(
                (d for d in runs_dir.iterdir() if d.is_dir()),
                key=lambda d: int(d.name.rsplit("_", 1)[-1]) if d.name.rsplit("_", 1)[-1].isdigit() else 0,
                reverse=True,
            )
            for run_dir in candidates:
                w = run_dir / "weights" / "best.pt"
                if w.exists():
                    return str(w)
        return None

    _weights = _latest_weights_for(_model_type)
    if not _weights:
        print(f"ERROR: no baseline weights found for {_model_type.value} — run baseline first")
        sys.exit(1)

    print(f"Model:   {_model_type.value}")
    print(f"Weights: {_weights}")

    with get_session() as _session:
        _run = TrainingRun(
            stage=TrainingStage.FINETUNE,
            model_type=_model_type,
            status=TrainingStatus.QUEUED,
            baseline_weights=_weights,
            created_at=datetime.utcnow(),
        )
        _session.add(_run)
        _session.flush()
        _run_id = _run.id
        print(f"Created TrainingRun id={_run_id}")

    os.chdir(Path(__file__).parent.parent)
    train_finetune.run(training_run_id=_run_id)
