import hashlib
import re
from datetime import datetime, timezone
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

_ANNOTATED_DIR = Path(__file__).parent.parent.parent.parent / "ml-engine" / "media" / "annotated"
_RAW_DIR       = Path(__file__).parent.parent.parent.parent / "scraper-engine" / "media" / "raw"

_MODEL_RE = re.compile(r'(aircraft|vehicle|personnel|general)', re.I)

_SOURCE_DISPLAY = {
    "funker530":    "Funker530",
    "geoconfirmed": "GeoConfirmed",
    "submitted":    "Submitted",
}


def _lookup_source(hash_prefix: str) -> str:
    """Find source by checking which raw subfolder contains a clip with this hash prefix."""
    for folder in _RAW_DIR.iterdir():
        if not folder.is_dir():
            continue
        if any(folder.glob(f"{hash_prefix}*")):
            return _SOURCE_DISPLAY.get(folder.name.lower(), folder.name.capitalize())
    return "Unknown"


def _video_duration(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    secs = int(frames / fps) if fps > 0 else 0
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@router.get("/annotated-clips")
async def get_annotated_clips() -> list[dict]:
    items = []
    for f in sorted(_ANNOTATED_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        hash_prefix = f.stem[:8]
        model_match = _MODEL_RE.search(f.stem)
        det_class   = model_match.group().upper() if model_match else "AIRCRAFT"
        mtime       = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        items.append({
            "id":       f.stem,
            "title":    f.stem.replace("_", " ").replace("annotated", "").strip().upper(),
            "date":     mtime.strftime("%Y-%m-%d"),
            "duration": _video_duration(f),
            "detClass": det_class,
            "source":   _lookup_source(hash_prefix),
            "tag":      "annotated",
            "src":      hash_prefix.upper(),
            "videoUrl": f"/media/annotated/{f.name}",
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

        stmt = (
            select(TrainingRun)
            .where(TrainingRun.model_type == mtype)
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )
        run = (await db.execute(stmt)).scalar_one_or_none()

        if run is None:
            result[model] = {"status": "QUEUED", "map50": None, "images": 0}
            continue

        metrics = run.metrics or {}
        images = metrics.get("total_train_images") or 0

        if run.status == TrainingStatus.DONE:
            key = next((k for k in metrics if "map50" in k.lower() and "map50-95" not in k.lower()), None)
            map50 = None
            if key:
                try: map50 = round(float(metrics[key]), 3)
                except (ValueError, TypeError): pass
            result[model] = {"status": "DONE", "map50": map50, "images": images}
        elif run.status == TrainingStatus.RUNNING:
            result[model] = {"status": "TRAINING", "map50": _live_map50(model), "images": images}
        elif run.status == TrainingStatus.ERROR:
            result[model] = {"status": "ERROR", "map50": _live_map50(model), "images": images}
        else:
            result[model] = {"status": "QUEUED", "map50": None, "images": images}

    return result


def _dir_gb(base: Path) -> float:
    import os
    total = 0
    for root, _dirs, files in _os_walk(base):
        for f in files:
            try:
                total += os.path.getsize(root / f)
            except OSError:
                pass
    return round(total / 1e9, 2)


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)) -> dict:
    # Count raw clips from disk — DB may be out of sync with directly-downloaded clips
    clips_total = 0
    if _RAW_DIR.exists():
        for src_dir in _RAW_DIR.iterdir():
            if src_dir.is_dir():
                clips_total += len(list(src_dir.glob("*.mp4")))

    annotated_mp4s = len(list(_ANNOTATED_DIR.glob("*.mp4"))) if _ANNOTATED_DIR.exists() else 0
    raw_gb = _dir_gb(_RAW_DIR) if _RAW_DIR.exists() else 0.0

    images_labeled = 0
    if _KAGGLE_DIR.exists():
        for ds_dir in _KAGGLE_DIR.iterdir():
            if ds_dir.name in _SOURCE_DATASETS and ds_dir.is_dir():
                images_labeled += _count_images(ds_dir)

    return {
        "clips_total":     clips_total,
        "clips_annotated": annotated_mp4s,
        "raw_gb":          raw_gb,
        "images_labeled":  images_labeled,
        "annotated_mp4s":  annotated_mp4s,
        "models":          await _model_stats(db),
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
        status=ClipStatus.PENDING,
    )
    db.add(clip)
    await db.flush()
    return ClipOut.model_validate(clip)
