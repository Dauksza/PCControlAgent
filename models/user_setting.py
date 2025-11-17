"""SQLAlchemy model for persisted user settings."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.utcnow()


class UserSetting(Base):
    """Generic key/value settings store."""

    __tablename__ = "user_settings"
    __table_args__ = (UniqueConstraint("setting_name", name="uq_setting_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    setting_name: Mapped[str] = mapped_column(String(255), nullable=False)
    setting_value: Mapped[str] = mapped_column(String, nullable=False)
    encrypted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "setting_name": self.setting_name,
            "encrypted": self.encrypted,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
