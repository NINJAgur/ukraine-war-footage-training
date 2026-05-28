"""
inference-engine/tasks/weights.py

Shared weight-resolution helpers used by annotate_clips and package_dataset.
"""
from pathlib import Path

from db.models import ModelType, TrainingRun, TrainingStatus
from db.session import get_session

INFERENCE_ENGINE_DIR = Path(__file__).resolve().parents[1]
TRAINING_ENGINE_DIR = INFERENCE_ENGINE_DIR.parent / "training-engine"


def _resolve_weights_path(raw: str) -> Path:
    """Handle Windows absolute paths or legacy ml-engine paths stored in DB."""
    w = Path(raw)
    if w.exists():
        return w
    normalized = raw.replace("\\", "/")
    for marker, strip_len in (
        ("training-engine/runs/", len("training-engine/")),
        ("ml-engine/runs/", len("ml-engine/")),
    ):
        if marker in normalized:
            rel = normalized[normalized.index(marker) + strip_len:]
            return TRAINING_ENGINE_DIR / rel
    return w


def _best_map50(metrics: dict) -> float:
    if not metrics:
        return 0.0
    for key in ("mAP50(B)", "metrics/mAP50(B)", "mAP50"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _latest_weights(model_name: str) -> Path:
    """Return the best-mAP50 weights path for model_name from the DB."""
    with get_session() as session:
        runs = (
            session.query(TrainingRun)
            .filter(
                TrainingRun.model_type == ModelType[model_name],
                TrainingRun.status == TrainingStatus.DONE,
                TrainingRun.weights_path.isnot(None),
            )
            .all()
        )
        ranked = sorted(runs, key=lambda r: _best_map50(r.metrics), reverse=True)
        paths = [(r.weights_path, _best_map50(r.metrics)) for r in ranked]
    for weights_path, _ in paths:
        w = _resolve_weights_path(weights_path)
        if w.exists():
            return w
    raise FileNotFoundError(f"No usable weights for {model_name} in DB")
