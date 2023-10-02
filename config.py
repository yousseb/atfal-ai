from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

env_location = Path("./.env").resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_location, env_file_encoding="utf-8")
    API_KEYS: set[str]
