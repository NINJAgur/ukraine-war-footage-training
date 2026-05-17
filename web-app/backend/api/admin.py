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
        queue="gpu",
        ignore_result=True,
    )
    return {"task_id": task.id, "training_run_id": run.id, "status": "QUEUED"}
