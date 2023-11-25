import logging
from pathlib import Path

from pydantic_settings import SettingsConfigDict

from app.core.settings.app import AppSettings

env_location = Path(".env").resolve()


class DevAppSettings(AppSettings):
    debug: bool = True
    logging_level: int = logging.DEBUG

    title: str = "Dev Atfalmafkoda AI APIs 🚀"
    model_config = SettingsConfigDict(env_file=env_location, env_file_encoding="utf-8")
