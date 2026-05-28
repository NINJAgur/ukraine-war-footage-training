"""
One-time backfill: set file_size_bytes for ANNOTATED clips where it is NULL.
Run inside the backend container:
  docker exec backend python /app/web-app/backend/scripts/backfill_sizes.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "/app/web-app/backend")
from db.session import AsyncSessionLocal
from sqlalchemy import text

ANNOTATED_DIR = Path("/app/inference-engine/media")


def _resolve(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    n = raw.replace("\\", "/")
    for marker, slen in (
        ("inference-engine/media/", len("inference-engine/media/")),
        ("ml-engine/media/annotated/", len("ml-engine/media/annotated/")),
    ):
        if marker in n:
            return ANNOTATED_DIR / n[n.index(marker) + slen :]
    return p


async def backfill() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT id, mp4_path FROM clips WHERE status='ANNOTATED' AND file_size_bytes IS NULL")
        )
        rows = result.fetchall()
        updated = skipped = 0
        for clip_id, mp4_path in rows:
            if not mp4_path or mp4_path.startswith("http"):
                skipped += 1
                continue
            p = _resolve(mp4_path)
            if p.exists():
                size = p.stat().st_size
                await session.execute(
                    text("UPDATE clips SET file_size_bytes = :sz WHERE id = :id"),
                    {"sz": size, "id": clip_id},
                )
                updated += 1
            else:
                skipped += 1
        await session.commit()
        print(f"Backfilled {updated}/{len(rows)} clips ({skipped} skipped — file not found or remote URL)")


if __name__ == "__main__":
    asyncio.run(backfill())
