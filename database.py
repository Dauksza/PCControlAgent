"""SQLite database configuration for conversation persistence."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from utils.logging_config import get_logger

logger = get_logger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR / 'conversations.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    """Create database tables if they do not exist."""
    try:
        import models.conversation  # noqa: F401  # Ensure models are registered
        import models.user_setting  # noqa: F401

        Base.metadata.create_all(bind=engine)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Database initialization failed: %s", exc)
        raise


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Initialize tables at import time for convenience.
init_db()
