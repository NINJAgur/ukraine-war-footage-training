from datetime import datetime
from typing import Optional

from celery import Celery
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_admin
from config import settings
from db.models import Clip, ClipStatus, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_db
from schemas.clips import ClipOut
from schemas.training import TrainRequest, TrainingRunOut

router = APIRouter(prefix="/api/admin", tags=["admin"])

_celery = Celery(broker=settings.REDIS_URL)


@router.get("/clips")
async def list_clips(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[ClipStatus] = None,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(get_current_admin),
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


@router.get("/training-runs")
async def list_training_runs(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(get_current_admin),
) -> dict:
    offset = (page - 1) * per_page
    total = (await db.execute(select(func.count()).select_from(TrainingRun))).scalar()
    items = (await db.execute(
        select(TrainingRun).order_by(TrainingRun.created_at.desc()).offset(offset).limit(per_page)
    )).scalars().all()
    return {"items": [TrainingRunOut.model_validate(r) for r in items], "total": total, "page": page, "per_page": per_page}


@router.post("/clips/{clip_id}/approve", status_code=200)
async def approve_clip(
    clip_id: int,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(get_current_admin),
) -> dict:
    clip = await db.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if clip.status != ClipStatus.REVIEW:
        raise HTTPException(status_code=409, detail=f"Clip is {clip.status}, not REVIEW")
    clip.status = ClipStatus.PENDING
    return {"clip_id": clip_id, "status": "PENDING"}


@router.delete("/clips/{clip_id}", status_code=200)
async def decline_clip(
    clip_id: int,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(get_current_admin),
) -> dict:
    clip = await db.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if clip.status != ClipStatus.REVIEW:
        raise HTTPException(status_code=409, detail=f"Clip is {clip.status}, not REVIEW — only submitted clips can be declined")
    await db.delete(clip)
    return {"clip_id": clip_id, "status": "DECLINED"}


@router.post("/train", status_code=202)
async def start_training(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(get_current_admin),
) -> dict:
    # Block if a run is already active for this model
    active = (await db.execute(
        select(TrainingRun)
        .where(TrainingRun.model_type == body.model_type)
        .where(TrainingRun.status.in_([TrainingStatus.QUEUED, TrainingStatus.RUNNING]))
        .limit(1)
    )).scalar_one_or_none()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"{body.model_type.value} already has a {active.status.value} run (id={active.id})",
        )

    # FINETUNE requires a completed baseline with weights on disk
    if body.stage == TrainingStage.FINETUNE:
        baseline = (await db.execute(
            select(TrainingRun)
            .where(TrainingRun.model_type == body.model_type)
            .where(TrainingRun.stage == TrainingStage.BASELINE)
            .where(TrainingRun.status == TrainingStatus.DONE)
            .where(TrainingRun.weights_path.isnot(None))
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )).scalar_one_or_none()
        if not baseline:
            raise HTTPException(
                status_code=400,
                detail=f"No completed baseline run found for {body.model_type.value} — run BASELINE first",
            )
    run = TrainingRun(
        stage=body.stage,
        model_type=body.model_type,
        status=TrainingStatus.QUEUED,
        baseline_weights=baseline.weights_path if body.stage == TrainingStage.FINETUNE else None,
        created_at=datetime.utcnow(),
    )
    db.add(run)
    await db.flush()
    task_name = "tasks.train_baseline.train_baseline" if body.stage == TrainingStage.BASELINE else "tasks.train_finetune.train_finetune"
    task = _celery.send_task(
        task_name,
        kwargs={"training_run_id": run.id},
        queue="training",
        ignore_result=True,
    )
    return {"task_id": task.id, "training_run_id": run.id, "status": "QUEUED"}


@router.get("/scraper-stats")
async def get_scraper_stats(
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(get_current_admin),
) -> dict:
    """Scraper pipeline counts by status and source."""
    from sqlalchemy import func, select
    from db.models import Clip, Dataset

    # Dataset pipeline counts by status
    dataset_rows = (await db.execute(
        select(Dataset.status, func.count(Dataset.id).label("count"))
        .group_by(Dataset.status)
    )).all()
    dataset_counts = {r.status.value if r.status else "unknown": r.count for r in dataset_rows}

    # PACKAGED datasets per model type (training triggers on image count threshold, not dataset count)
    from sqlalchemy import text as sa_text
    packaged_per_model = {}
    for model in ("AIRCRAFT", "VEHICLE", "PERSONNEL", "GENERAL"):
        if model == "GENERAL":
            count_row = (await db.execute(
                select(func.count(Dataset.id))
                .where(Dataset.status == "PACKAGED")
            )).scalar()
        else:
            count_row = (await db.execute(
                sa_text(
                    f"SELECT COUNT(*) FROM datasets WHERE status='PACKAGED' "
                    f"AND detected_model_types::text LIKE '%{model}%'"
                )
            )).scalar()
        packaged_per_model[model] = count_row or 0

    # Counts by status
    status_rows = (await db.execute(
        select(Clip.status, func.count(Clip.id).label("count"))
        .group_by(Clip.status)
    )).all()
    by_status = {r.status.value if r.status else "unknown": r.count for r in status_rows}

    # Counts by source
    source_rows = (await db.execute(
        select(Clip.source, func.count(Clip.id).label("count"))
        .group_by(Clip.source)
    )).all()
    by_source = {r.source.value if r.source else "unknown": r.count for r in source_rows}

    # Recent clips (last 5 ingested)
    recent = (await db.execute(
        select(Clip).order_by(Clip.created_at.desc()).limit(5)
    )).scalars().all()
    recent_list = [{"title": (c.title or c.url_hash[:12])[:50], "status": c.status.value if c.status else None, "source": c.source.value if c.source else None, "created_at": c.created_at.isoformat() if c.created_at else None} for c in recent]

    return {
        "total": sum(by_status.values()),
        "by_status": by_status,
        "by_source": by_source,
        "dataset_pipeline": dataset_counts,
        "packaged_per_model": packaged_per_model,
        "packaged_detail": [
            {"id": r.id, "models": r.detected_model_types}
            for r in (await db.execute(
                select(Dataset.id, Dataset.detected_model_types)
                .where(Dataset.status == "PACKAGED")
                .order_by(Dataset.id)
            )).all()
        ],
        "recent": recent_list,
    }


