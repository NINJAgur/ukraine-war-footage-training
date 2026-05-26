import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Clip, ClipSource, ClipStatus, ModelType, TrainingRun, TrainingStatus
from db.session import get_db
from schemas.clips import ClipOut, ClipSubmit

router = APIRouter(prefix="/api", tags=["public"])

_ANNOTATED_DIR = Path(__file__).parent.parent.parent.parent / "ml-engine" / "media" / "annotated"


def _resolve_mp4_path(raw: str) -> Path:
    """Translate Windows absolute paths stored in DB to container/local paths."""
    p = Path(raw)
    if p.exists():
        return p
    normalized = raw.replace("\\", "/")
    marker = "ml-engine/media/annotated/"
    if marker in normalized:
        rel = normalized[normalized.index(marker) + len(marker):]
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
            "id":       clip.url_hash[:12],
            "title":    (clip.title or clip.url_hash[:12]).upper(),
            "date":     clip.created_at.strftime("%Y-%m-%d") if clip.created_at else "",
            "duration": duration_str,
            "detClass": clip.det_class,
            "source":   _SOURCE_DISPLAY.get((clip.source.value if clip.source else ""), "Unknown"),
            "tag":      "annotated",
            "src":      clip.url_hash[:8].upper(),
            "videoUrl": video_url,
        })
    return items


_KAGGLE_DIR  = Path(__file__).parent.parent.parent.parent / "ml-engine" / "media" / "kaggle_datasets"
_RUNS_DIR    = Path(__file__).parent.parent.parent.parent / "ml-engine" / "runs" / "baseline"
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
    result = {}
    for model in _MODELS:
        try:
            mtype = ModelType[model]
        except KeyError:
            result[model] = {"status": "QUEUED", "map50": None, "images": 0}
            continue

        # Best completed run by mAP50 — source of truth for mAP and image count
        done_runs = (await db.execute(
            select(TrainingRun)
            .where(TrainingRun.model_type == mtype, TrainingRun.status == TrainingStatus.DONE)
        )).scalars().all()

        def _run_map50(r: TrainingRun) -> float:
            m = r.metrics or {}
            k = next((k for k in m if "map50" in k.lower() and "map50-95" not in k.lower()), None)
            try: return float(m[k]) if k else 0.0
            except (ValueError, TypeError): return 0.0

        done_run = max(done_runs, key=_run_map50) if done_runs else None

        # Active run — determines displayed status
        active_run = (await db.execute(
            select(TrainingRun)
            .where(TrainingRun.model_type == mtype, TrainingRun.status == TrainingStatus.RUNNING)
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )).scalar_one_or_none()

        if done_run is None and active_run is None:
            result[model] = {"status": "QUEUED", "map50": None, "images": 0}
            continue

        # mAP and images always come from the best DONE run
        map50 = None
        images = 0
        if done_run:
            metrics = done_run.metrics or {}
            images = metrics.get("total_train_images") or 0
            key = next((k for k in metrics if "map50" in k.lower() and "map50-95" not in k.lower()), None)
            if key:
                try: map50 = round(float(metrics[key]), 3)
                except (ValueError, TypeError): pass

        if active_run:
            result[model] = {"status": "TRAINING", "map50": map50, "images": images}
        else:
            result[model] = {"status": "DONE", "map50": map50, "images": images}

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