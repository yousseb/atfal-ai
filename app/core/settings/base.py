from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvTypes(Enum):
    prod: str = "prod"
    dev: str = "dev"
    test: str = "test"


class BaseAppSettings(BaseSettings):
    app_env: AppEnvTypes = AppEnvTypes.prod
    model_config = SettingsConfigDict(extra='ignore', env_file=".env", env_file_encoding='utf-8')
