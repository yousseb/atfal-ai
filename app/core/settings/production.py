from pathlib import Path

from pydantic_settings import SettingsConfigDict

from app.core.settings.app import AppSettings

env_prod_location = Path("/config/.env.prod").resolve()


class ProdAppSettings(AppSettings):
    model_config = SettingsConfigDict(
        # `.env.prod` takes priority over `.env`
        env_file=(env_prod_location, env_prod_location),
        env_file_encoding="utf-8")
