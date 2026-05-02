import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.auth import router as auth_router
from api.public import router as public_router
from api.admin import router as admin_router
from config import settings

app = FastAPI(title="Ukraine Combat Footage API", version="0.1.0")

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

_ANNOTATED_DIR = Path(__file__).parent.parent.parent / "ml-engine" / "media" / "annotated"
_ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media/annotated", StaticFiles(directory=str(_ANNOTATED_DIR)), name="annotated")
