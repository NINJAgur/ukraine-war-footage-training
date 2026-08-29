import sys
from pathlib import Path

# Must be before any router imports so hot-reload picks up shared.db.models
_REPO_ROOT = str(Path(__file__).parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.auth import router as auth_router
from api.public import router as public_router
from api.admin import router as admin_router
from api.ws import router as ws_router
from config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from sqlalchemy import create_engine
    from shared.db.models import Base, sync_enum_values
    engine = create_engine(settings.DATABASE_SYNC_URL)
    Base.metadata.create_all(bind=engine)
    sync_enum_values(engine)
    engine.dispose()
    yield


app = FastAPI(title="Ukraine Combat Footage API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(public_router)
app.include_router(admin_router)
app.include_router(ws_router)

_ANNOTATED_DIR = Path(__file__).parent.parent.parent / "inference-engine" / "media"
_ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/media/annotated/{path:path}")
async def serve_annotated(path: str):
    # Resolve before testing containment — "%2e%2e%2f" survives URL decoding and
    # would otherwise walk out of the media directory.
    root = _ANNOTATED_DIR.resolve()
    f = (root / path).resolve()
    if not f.is_relative_to(root) or not f.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(f, media_type="video/mp4")
