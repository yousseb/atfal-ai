from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os

# DATABASE_FILE = "auth_db.sqlite"
# ENGINE_URI = "sqlite:///" + DATABASE_FILE
# SECRET = "1234567890"
#
# AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID", "your-auth0-client-id")
# AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET", "your-auth0-client-secret")
# AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "your-auth0-domain")

env_location = Path("./.env").resolve()
env_prod_location = Path("/config/.env.prod").resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # `.env.prod` takes priority over `.env`
        env_file=(env_location, env_prod_location),
        env_file_encoding="utf-8")
    API_KEYS: set[str]


@lru_cache()
def get_settings():
    return Settings()
