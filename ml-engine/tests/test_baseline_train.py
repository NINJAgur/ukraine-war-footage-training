"""
ml-engine/tests/test_baseline_train.py

Smoke-test for Stage 1 baseline training.
Creates a TrainingRun record, calls train_baseline() directly (no Celery),
verifies best.pt is produced, then optionally cleans up.

Run from ml-engine/:
    python tests/test_baseline_train.py --model-type VEHICLE --epochs 2
    python tests/test_baseline_train.py --model-type GENERAL --epochs 50 --keep
    python tests/test_baseline_train.py --model-type GENERAL --purge-outputs
"""
import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from db.models import ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("baseline-test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="VEHICLE",
                        choices=[m.value for m in ModelType])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--keep", action="store_true",
                        help="Keep weights and DB row after test")
    parser.add_argument("--purge-outputs", action="store_true",
                        help="Delete weights dir AND DB row after test")
    parser.add_argument("--weights", default=None,
                        help="Starting weights path (e.g. runs/baseline/AIRCRAFT/.../best.pt)")
    args = parser.parse_args()

    model_type = ModelType(args.model_type)

    # Patch epoch count for this test run
    original_epochs = settings.YOLO_EPOCHS_BASELINE
    settings.YOLO_EPOCHS_BASELINE = args.epochs
    logger.info(f"Epochs overridden to {args.epochs} for smoke test")

    run_id = None
    weights_path = None
    try:
        # Create TrainingRun record
        with get_session() as session:
            run = TrainingRun(
                stage=TrainingStage.BASELINE,
                model_type=model_type,
                status=TrainingStatus.QUEUED,
                created_at=datetime.utcnow(),
            )
            session.add(run)
            session.flush()
            run_id = run.id
        logger.info(f"Created TrainingRun id={run_id} model_type={model_type.value}")

        # Call task directly (bypasses Celery)
        from tasks.train_baseline import train_baseline
        result = train_baseline(training_run_id=run_id, weights=args.weights)
        logger.info(f"Result: {result}")

        assert result["status"] == "done", f"Expected done, got {result['status']}"
        weights_path = Path(result["weights_path"])
        assert weights_path.exists(), f"best.pt not found: {weights_path}"

        logger.info(f"✓ best.pt produced: {weights_path}  ({weights_path.stat().st_size // 1024}KB)")
        logger.info(f"✓ Metrics: {result.get('metrics', {})}")
        logger.info("BASELINE TRAIN TEST PASSED")

    except Exception as exc:
        logger.error(f"TEST FAILED: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        settings.YOLO_EPOCHS_BASELINE = original_epochs
        cleanup = args.purge_outputs or (not args.keep and not args.purge_outputs)
        if cleanup and run_id is not None:
            with get_session() as session:
                run = session.get(TrainingRun, run_id)
                if run:
                    session.delete(run)
            if args.purge_outputs and weights_path:
                run_dir = weights_path.parent.parent  # .../baseline_MODEL_N/
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)
                    logger.info(f"Removed weights dir: {run_dir}  (--purge-outputs)")
            elif not args.keep and weights_path:
                run_dir = weights_path.parent.parent  # .../baseline_MODEL_N/
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)
                    logger.info(f"Cleaned up run dir: {run_dir}")
            logger.info("Teardown complete.")


if __name__ == "__main__":
    main()
