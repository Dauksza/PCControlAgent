"""Helpers for managing user-configurable settings with encryption."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.exc import SQLAlchemyError

from database import session_scope
from models.user_setting import UserSetting
from utils.logging_config import get_logger

logger = get_logger(__name__)

SETTINGS_API_KEY = "mistral_api_key"
_KEY_FILE = Path("data") / "encryption.key"


def _load_encryption_key() -> bytes:
    """Read or generate the symmetric key used for encrypting secrets."""
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes()

    key = Fernet.generate_key()
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_bytes(key)
    return key


def _get_cipher() -> Fernet:
    return Fernet(_load_encryption_key())


def save_api_key(api_key: str) -> None:
    """Encrypt and persist the supplied API key."""
    cipher = _get_cipher()
    encrypted_value = cipher.encrypt(api_key.encode("utf-8")).decode("utf-8")

    try:
        with session_scope() as session:
            setting = (
                session.query(UserSetting)
                .filter(UserSetting.setting_name == SETTINGS_API_KEY)
                .one_or_none()
            )
            if setting:
                setting.setting_value = encrypted_value
                setting.encrypted = True
            else:
                session.add(
                    UserSetting(
                        setting_name=SETTINGS_API_KEY,
                        setting_value=encrypted_value,
                        encrypted=True,
                    )
                )
    except SQLAlchemyError as exc:
        logger.error("Failed to save API key: %s", exc)
        raise


def get_api_key() -> Optional[str]:
    """Retrieve and decrypt the stored API key if present."""
    try:
        with session_scope() as session:
            setting = (
                session.query(UserSetting)
                .filter(UserSetting.setting_name == SETTINGS_API_KEY)
                .one_or_none()
            )
            if not setting:
                return None

            if not setting.encrypted:
                return setting.setting_value

            cipher = _get_cipher()
            try:
                decrypted = cipher.decrypt(setting.setting_value.encode("utf-8"))
                return decrypted.decode("utf-8")
            except InvalidToken:
                logger.error("Stored API key could not be decrypted; removing corrupted value.")
                session.delete(setting)
                return None
    except SQLAlchemyError as exc:
        logger.error("Failed to retrieve API key: %s", exc)
        raise


def delete_api_key() -> bool:
    """Delete the stored API key, returning True if a value was removed."""
    try:
        with session_scope() as session:
            setting = (
                session.query(UserSetting)
                .filter(UserSetting.setting_name == SETTINGS_API_KEY)
                .one_or_none()
            )
            if not setting:
                return False
            session.delete(setting)
            return True
    except SQLAlchemyError as exc:
        logger.error("Failed to delete API key: %s", exc)
        raise
