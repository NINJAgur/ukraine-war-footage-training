"""
scraper-engine/db/session.py
Synchronous SQLAlchemy session factory for Celery tasks.

Celery workers are synchronous; asyncpg/AsyncSession is for FastAPI only.
"""
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Base

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_SYNC_URL,
    pool_pre_ping=True,       # verify connection health before use
    pool_size=5,              # max persistent connections
    max_overflow=10,          # additional connections allowed under load
    echo=False,               # set True for SQL debug logging
)

# ── Session factory ───────────────────────────────────────────────────
SyncSessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,   # keep object attributes accessible after commit
)


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    logger.info("Initializing database schema…")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready.")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager providing a transactional database session.

    Usage:
        with get_session() as session:
            clip = session.query(Clip).filter_by(url_hash=h).first()
            session.add(new_clip)
        # auto-committed on exit; rolled back on exception
    """
    session: Session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
