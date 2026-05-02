import hashlib
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Clip, ClipSource, ClipStatus
from db.session import get_db
from schemas.clips import ClipOut, ClipSubmit

router = APIRouter(prefix="/api", tags=["public"])


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
