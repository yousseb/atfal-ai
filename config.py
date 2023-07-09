from pydantic import BaseSettings

# TODO: Actually use settings...


class Settings(BaseSettings):
    app_name: str = "Atfalmafkoda API"
    admin_email: str

    class Config:
        env_file = ".env"
