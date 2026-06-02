import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Clip, ClipSource, ClipStatus, ModelType, TrainingRun, TrainingStatus
from db.session import get_db
from schemas.clips import ClipOut, ClipSubmit

router = APIRouter(prefix="/api", tags=["public"])

_ANNOTATED_DIR = Path(__file__).parent.parent.parent.parent / "inference-engine" / "media"


def _resolve_mp4_path(raw: str) -> Path:
    """Translate Windows absolute paths stored in DB to container/local paths."""
    p = Path(raw)
    if p.exists():
        return p
    normalized = raw.replace("\\", "/")
    for marker, strip_len in (
        ("inference-engine/media/", len("inference-engine/media/")),
        ("ml-engine/media/", len("ml-engine/media/")),
    ):
        if marker in normalized:
            rel = normalized[normalized.index(marker) + strip_len:]
            return _ANNOTATED_DIR / rel
    return p

_SOURCE_DISPLAY = {
    "funker530":    "Funker530",
    "geoconfirmed": "GeoConfirmed",
    "submitted":    "Submitted",
}


def _video_duration(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    secs = int(frames / fps) if fps > 0 else 0
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@router.get("/annotated-clips")
async def get_annotated_clips(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Return annotated clips from DB — det_class is authoritative, never derived from filename."""
    stmt = (
        select(Clip)
        .where(Clip.status == ClipStatus.ANNOTATED, Clip.mp4_path.isnot(None), Clip.det_class.isnot(None))
        .order_by(Clip.created_at.desc())
    )
    clips = (await db.execute(stmt)).scalars().all()
    items = []
    
    for clip in clips:
        # Support both local paths and remote URLs
        if clip.mp4_path.startswith("http"):
            video_url = clip.mp4_path
            # For remote files, rely on DB duration rather than OpenCV scanning
            duration_str = f"00:00:{clip.duration_seconds:02d}" if clip.duration_seconds else "00:00:00"
        else:
            mp4 = _resolve_mp4_path(clip.mp4_path)
            if not mp4.exists():
                continue
            # videoUrl uses the category subdir path
            rel = mp4.relative_to(_ANNOTATED_DIR)
            video_url = f"/media/annotated/{rel.as_posix()}"
            duration_str = _video_duration(mp4)

        items.append({
            "id":          clip.url_hash[:12],
            "title":       (clip.title or clip.url_hash[:12]).upper(),
            "description": clip.description or "",
            "date":        clip.created_at.strftime("%Y-%m-%d") if clip.created_at else "",
            "duration":    duration_str,
            "detClass":    clip.det_class,
            "source":      _SOURCE_DISPLAY.get((clip.source.value if clip.source else ""), "Unknown"),
            "tag":      "annotated",
            "src":      clip.url_hash[:8].upper(),
            "videoUrl": video_url,
        })
    return items


_KAGGLE_DIR  = Path(__file__).parent.parent.parent.parent / "training-engine" / "media" / "kaggle_datasets"
_RUNS_DIR    = Path(__file__).parent.parent.parent.parent / "training-engine" / "runs" / "baseline"
_ALL_RUNS_DIR = Path(__file__).parent.parent.parent.parent / "training-engine" / "runs"
_SOURCE_DATASETS = {"mihprofi", "nzigulic", "piterfm", "shakedlevnat", "sudipchakrabarty"}
_MODELS = ["AIRCRAFT", "VEHICLE", "PERSONNEL", "GENERAL"]


def _count_images(base: Path) -> int:
    total = 0
    for root, _dirs, files in base.walk() if hasattr(base, 'walk') else _os_walk(base):
        total += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return total


def _os_walk(base: Path):
    import os
    for root, dirs, files in os.walk(str(base)):
        yield Path(root), dirs, files


def _live_map50(model: str) -> Optional[float]:
    import csv as _csv
    mdir = _RUNS_DIR / model
    if not mdir.exists():
        return None
    runs = sorted(d.name for d in mdir.iterdir() if d.is_dir())
    if not runs:
        return None
    csv_path = mdir / runs[-1] / "results.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return None
    key = next((k for k in rows[-1] if "map50" in k.lower() and "map50-95" not in k.lower()), None)
    if not key:
        return None
    try:
        return round(float(rows[-1][key].strip()), 3)
    except ValueError:
        return None


async def _model_stats(db: AsyncSession) -> dict:
    # Single query: fetch all training runs in one shot
    all_runs = (await db.execute(
        select(TrainingRun).where(TrainingRun.status.in_([TrainingStatus.DONE, TrainingStatus.RUNNING]))
    )).scalars().all()

    def _run_map50(r: TrainingRun) -> float:
        m = r.metrics or {}
        k = next((k for k in m if "map50" in k.lower() and "map50-95" not in k.lower()), None)
        try:
            return float(m[k]) if k else 0.0
        except (ValueError, TypeError):
            return 0.0

    result = {}
    for model in _MODELS:
        try:
            mtype = ModelType[model]
        except KeyError:
            result[model] = {"status": "QUEUED", "map50": None, "images": 0}
            continue

        done_runs = [r for r in all_runs if r.model_type == mtype and r.status == TrainingStatus.DONE]
        active_run = next((r for r in all_runs if r.model_type == mtype and r.status == TrainingStatus.RUNNING), None)
        done_run = max(done_runs, key=lambda r: r.completed_at or r.created_at) if done_runs else None

        if done_run is None and active_run is None:
            result[model] = {"status": "QUEUED", "map50": None, "images": 0}
            continue

        map50 = None
        images = 0
        if done_run:
            metrics = done_run.metrics or {}
            images = metrics.get("total_train_images") or 0
            key = next((k for k in metrics if "map50" in k.lower() and "map50-95" not in k.lower()), None)
            if key:
                try:
                    map50 = round(float(metrics[key]), 3)
                except (ValueError, TypeError):
                    pass

        result[model] = {"status": "TRAINING" if active_run else "DONE", "map50": map50, "images": images}

    return result


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)) -> dict:
    rows = (await db.execute(
        select(
            func.count(Clip.id).label("total"),
            func.count(Clip.id).filter(Clip.status == ClipStatus.ANNOTATED).label("annotated"),
            func.coalesce(func.sum(Clip.file_size_bytes).filter(Clip.status == ClipStatus.ANNOTATED), 0).label("total_bytes"),
        )
    )).one()
    clips_total    = rows.total    or 0
    annotated_mp4s = rows.annotated or 0
    storage_gb     = round(int(rows.total_bytes or 0) / 1e9, 2)

    model_stats = await _model_stats(db)
    # Use GENERAL's image count as the canonical unique-images total — it already
    # contains all data from the specialist models, so summing all 4 would double-count.
    general = model_stats.get("GENERAL", {})
    images_labeled = general.get("images", 0) if general.get("status") == "DONE" else sum(
        m.get("images", 0) for m in model_stats.values()
        if m.get("status") == "DONE" and m.get("model_type") != "GENERAL"
    )

    return {
        "clips_total":     clips_total,
        "clips_annotated": annotated_mp4s,
        "raw_gb":          storage_gb,
        "images_labeled":  images_labeled,
        "annotated_mp4s":  annotated_mp4s,
        "models":          model_stats,
        "sources_active":  2,
    }


_GCS_BUCKET = "ukraine-footage-media"
_CLASS_DEFS = {
    "AIRCRAFT":  {"id": 0, "covers": "Drones, helicopters, fixed-wing aircraft, missiles, glide bombs"},
    "VEHICLE":   {"id": 1, "covers": "Tanks, APCs, artillery, radar, all ground military vehicles"},
    "PERSONNEL": {"id": 2, "covers": "Soldiers, fighters, RPG/ATGM operators"},
    "GENERAL":   {"id": None, "covers": "All three classes combined — AIRCRAFT + VEHICLE + PERSONNEL"},
}


def _run_to_dict(r: TrainingRun, model_name: str, is_best: bool) -> dict:
    metrics = r.metrics or {}
    map50_key = next((k for k in metrics if "map50" in k.lower() and "map50-95" not in k.lower()), None)
    map50 = round(float(metrics[map50_key]), 3) if map50_key else None
    images = metrics.get("total_train_images") or 0
    download_url = None
    if r.weights_path:
        wp = r.weights_path.replace("\\", "/")
        for marker in ("runs/finetune/", "runs/baseline/"):
            if marker in wp:
                gcs_key = wp[wp.index(marker):]
                download_url = f"https://storage.googleapis.com/{_GCS_BUCKET}/{gcs_key}"
                break
    return {
        "model":        model_name,
        "run_id":       r.id,
        "stage":        r.stage.value if r.stage else "FINETUNE",
        "map50":        map50,
        "images":       images,
        "download_url": download_url,
        "classes":      _CLASS_DEFS.get(model_name, {}),
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "is_best":      is_best,
    }


@router.get("/models")
async def get_models(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Return all completed training runs with is_best flag per model type."""
    all_done = (await db.execute(
        select(TrainingRun)
        .where(TrainingRun.status == TrainingStatus.DONE)
        .order_by(TrainingRun.completed_at.desc())
    )).scalars().all()

    result = []
    for model_name in _MODELS:
        try:
            mtype = ModelType[model_name]
        except KeyError:
            continue
        done_runs = sorted(
            [r for r in all_done if r.model_type == mtype],
            key=lambda r: r.completed_at or r.created_at,
            reverse=True,
        )
        if not done_runs:
            continue
        best_id = max(done_runs, key=lambda r: _run_map50(r) or 0).id
        for r in done_runs:
            result.append(_run_to_dict(r, model_name, r.id == best_id))
    return result


def _run_map50(r: TrainingRun) -> float:
    m = r.metrics or {}
    k = next((k for k in m if "map50" in k.lower() and "map50-95" not in k.lower()), None)
    try:
        return float(m[k]) if k else 0.0
    except (ValueError, TypeError):
        return 0.0


@router.get("/training/epoch-data")
async def get_epoch_data(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Return per-epoch training metrics for all completed runs.
    Reads from TrainingRun.metrics.epochs_data (set by train_finetune).
    Falls back to local results.csv files on dev."""
    import csv as _csv

    runs = (await db.execute(
        select(TrainingRun)
        .where(TrainingRun.status == TrainingStatus.DONE)
        .order_by(TrainingRun.id)
    )).scalars().all()

    result = []
    for r in runs:
        m = r.metrics or {}
        model = r.model_type.value if r.model_type else None
        stage = r.stage.value if r.stage else None
        epochs_data = m.get("epochs_data")
        confusion_matrix = m.get("confusion_matrix")

        # Dev fallback: read from local runs directory
        if not epochs_data and _ALL_RUNS_DIR.exists():
            run_name = f"{stage.lower()}_{model}_{r.id}" if stage and model else None
            if run_name:
                for csv_path in _ALL_RUNS_DIR.rglob(f"*{r.id}/results.csv"):
                    try:
                        with open(csv_path, newline="") as f:
                            epochs_data = [{k.strip(): float(v.strip()) for k, v in row.items() if v.strip()} for row in _csv.DictReader(f)]
                        break
                    except Exception:
                        pass

        if epochs_data:
            result.append({
                "run_id": r.id,
                "model": model,
                "stage": stage,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "epochs": epochs_data,
                "confusion_matrix": confusion_matrix,
            })

    return result


@router.get("/stats/charts")
async def get_stats_charts(
    days: int = Query(30, ge=7, le=90),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Aggregated chart data for the analytics section."""
    from datetime import timedelta
    from sqlalchemy import cast, Date as SADate

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Clips annotated per day
    clips_per_day_rows = (await db.execute(
        select(
            cast(Clip.created_at, SADate).label("day"),
            func.count(Clip.id).label("count"),
        )
        .where(Clip.status == ClipStatus.ANNOTATED, Clip.created_at >= cutoff)
        .group_by(cast(Clip.created_at, SADate))
        .order_by(cast(Clip.created_at, SADate))
    )).all()
    clips_per_day = [{"date": str(r.day), "count": r.count} for r in clips_per_day_rows]

    # Detection breakdown by det_class (time-filtered)
    breakdown_rows = (await db.execute(
        select(Clip.det_class, func.count(Clip.id).label("count"))
        .where(Clip.status == ClipStatus.ANNOTATED, Clip.det_class.isnot(None), Clip.created_at >= cutoff)
        .group_by(Clip.det_class)
    )).all()
    detection_breakdown = [{"class": r.det_class, "count": r.count} for r in breakdown_rows]

    # Total detection box counts per day — raw SQL for JSON extraction
    from sqlalchemy import cast, Date as SADate, text as sql_text
    try:
        det_rows = (await db.execute(
            sql_text("""
                SELECT
                    DATE(created_at) AS day,
                    COALESCE(SUM((detection_counts->>'aircraft')::int), 0) AS aircraft,
                    COALESCE(SUM((detection_counts->>'vehicle')::int),  0) AS vehicle,
                    COALESCE(SUM((detection_counts->>'personnel')::int),0) AS personnel
                FROM clips
                WHERE status = 'ANNOTATED'
                  AND detection_counts IS NOT NULL
                  AND created_at >= :cutoff
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """),
            {"cutoff": cutoff},
        )).all()
        detection_boxes_per_day = [{"date": str(r.day), "aircraft": int(r.aircraft), "vehicle": int(r.vehicle), "personnel": int(r.personnel)} for r in det_rows]
    except Exception:
        detection_boxes_per_day = []

    # All completed training runs (unfiltered for scatter/radar — full history needed)
    # mAP50 timeline filtered by cutoff
    runs = (await db.execute(
        select(TrainingRun)
        .where(TrainingRun.status == TrainingStatus.DONE, TrainingRun.completed_at.isnot(None))
        .order_by(TrainingRun.completed_at)
    )).scalars().all()
    runs_filtered = [r for r in runs if r.completed_at and r.completed_at >= cutoff]

    map50_timeline = []
    training_scatter = []  # all runs: {model, stage, map50, images, run_id}

    for r in runs_filtered:
        m = r.metrics or {}
        k50    = next((k for k in m if "map50" in k.lower() and "map50-95" not in k.lower()), None)
        k5095  = next((k for k in m if "map50-95" in k.lower()), None)
        kprec  = next((k for k in m if "precision" in k.lower()), None)
        krec   = next((k for k in m if "recall" in k.lower()), None)
        def _f(key, _m=m):
            try:
                return round(float(_m[key]), 3) if key and _m.get(key) else None
            except (ValueError, TypeError):
                return None
        map50  = _f(k50)
        images = m.get("total_train_images") or 0
        if map50 is not None:
            run_dict = {
                "run_id": r.id,
                "model": r.model_type.value if r.model_type else None,
                "stage": r.stage.value if r.stage else None,
                "map50": map50,
                "map50_95": _f(k5095),
                "precision": _f(kprec),
                "recall": _f(krec),
                "images": images,
                "date": r.completed_at.isoformat(),
            }
            map50_timeline.append(run_dict)
            training_scatter.append(run_dict)

    # Radar needs all-time best per model — add separately if not in filtered runs
    for r in runs:
        model_name2 = r.model_type.value if r.model_type else None
        if not model_name2: continue
        already = any(t["model"] == model_name2 for t in training_scatter)
        if not already:
            m2 = r.metrics or {}
            k2 = next((k for k in m2 if "map50" in k.lower() and "map50-95" not in k.lower()), None)
            try: v2 = round(float(m2[k2]), 3) if k2 else None
            except: v2 = None
            if v2:
                training_scatter.append({"run_id": r.id, "model": model_name2, "stage": r.stage.value if r.stage else None, "map50": v2, "map50_95": None, "precision": None, "recall": None, "images": m2.get("total_train_images") or 0, "date": r.completed_at.isoformat() if r.completed_at else None, "is_best": False})

    return {
        "clips_per_day": clips_per_day,
        "detection_breakdown": detection_breakdown,
        "detection_boxes_per_day": detection_boxes_per_day,
        "map50_timeline": map50_timeline,
        "training_scatter": training_scatter,
        "days": days,
    }


@router.get("/feed")
async def get_feed(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    offset = (page - 1) * per_page
    total = (await db.execute(
        select(func.count()).where(Clip.status == ClipStatus.ANNOTATED)
    )).scalar()
    items = (await db.execute(
        select(Clip)
        .where(Clip.status == ClipStatus.ANNOTATED)
        .order_by(Clip.created_at.desc())
        .offset(offset).limit(per_page)
    )).scalars().all()
    return {"items": [ClipOut.model_validate(c) for c in items], "total": total, "page": page, "per_page": per_page}


@router.get("/archive")
async def get_archive(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[ClipStatus] = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    offset = (page - 1) * per_page
    base = select(Clip)
    count_base = select(func.count()).select_from(Clip)
    if status:
        base = base.where(Clip.status == status)
        count_base = count_base.where(Clip.status == status)
    total = (await db.execute(count_base)).scalar()
    items = (await db.execute(
        base.order_by(Clip.created_at.desc()).offset(offset).limit(per_page)
    )).scalars().all()
    return {"items": [ClipOut.model_validate(c) for c in items], "total": total, "page": page, "per_page": per_page}


@router.post("/submit", response_model=ClipOut, status_code=201)
async def submit_clip(body: ClipSubmit, db: AsyncSession = Depends(get_db)) -> ClipOut:
    url_hash = hashlib.sha256(body.url.encode()).hexdigest()
    existing = (await db.execute(
        select(Clip).where(Clip.url_hash == url_hash)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Clip already exists")
    clip = Clip(
        url=body.url,
        url_hash=url_hash,
        title=body.title,
        description=body.description,
        source=ClipSource.SUBMITTED,
        status=ClipStatus.REVIEW,
    )
    db.add(clip)
    await db.flush()
    return ClipOut.model_validate(clip)