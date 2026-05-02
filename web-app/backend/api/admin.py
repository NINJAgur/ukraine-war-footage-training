from datetime import datetime
from typing import Optional

from celery import Celery
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_admin
from config import settings
from db.models import Clip, ClipStatus, TrainingRun, TrainingStatus
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


@router.post("/train", status_code=202)
async def start_training(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(get_current_admin),
) -> dict:
    run = TrainingRun(
        stage=body.stage,
        model_type=body.model_type,
        status=TrainingStatus.QUEUED,
        created_at=datetime.utcnow(),
    )
    db.add(run)
    await db.flush()
    task = _celery.send_task(
        "tasks.train_baseline.train_baseline",
        kwargs={"training_run_id": run.id},
        queue="gpu",
    )
    return {"task_id": task.id, "training_run_id": run.id, "status": "QUEUED"}
