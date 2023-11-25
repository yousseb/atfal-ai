
import logging
import sys
from typing import Any, Dict, List, Tuple

from loguru import logger
from pydantic import SecretStr

from app.core.logging import InterceptHandler
from app.core.settings.base import BaseAppSettings

description = """
This site should not be used except by Atfalmafkoda scheduled jobs.

Code running this API can be found here: <https://github.com/yousseb/atfal-ai>

## Faces

* Detect faces and returns **boxes** where faces could be within a given image.
* Enhances faces
* Frontalizes faces

"""


class AppSettings(BaseAppSettings, validate_assignment=True):
    debug: bool = False
    docs_url: str = "/docs"
    openapi_prefix: str = ""
    openapi_url: str = "/openapi.json"
    redoc_url: str = "/redoc"
    title: str = "Atfalmafkoda AI APIs 🚀"
    summary: str = "AI API engine to help extract and match missing cases."
    version: str = "0.0.5"
    contact: dict = {
        "name": "Atfalmafkoda",
        "url": "https://atfalmafkoda.com/",
    }
    license_info: dict = {
        "name": "MIT",
        "url": "https://opensource.org/license/mit/",
    }
    swagger_ui_parameters: dict = {
        'Bearer': {
            'type': 'apiKey',
            'name': 'Authorization',
            'in': 'header',
            'description': '<hr/>'
                           'Enter the word <tt>Token</tt> followed by space then your apiKey <br/><br/> '
                           '<b>Example:</b> <pre>Token f4bff35e0f6427860ae31bde0b5f2352cbf73d80</pre>'
                           '<hr/><br/>'
        }
    }

    # database_url: PostgresDsn
    # max_connection_count: int = 10
    # min_connection_count: int = 10
    # secret_key: SecretStr

    api_keys: str

    api_prefix: str = "/api"

    allowed_hosts: List[str] = ["*"]

    logging_level: int = logging.INFO
    loggers: Tuple[str, str] = ("uvicorn.asgi", "uvicorn.access")

    @property
    def fastapi_kwargs(self) -> Dict[str, Any]:
        return {
            "debug": self.debug,
            "docs_url": self.docs_url,
            "openapi_prefix": self.openapi_prefix,
            "openapi_url": self.openapi_url,
            "redoc_url": self.redoc_url,
            "title": self.title,
            "version": self.version,
            "summary": self.summary,
            "contact": self.contact,
            "license_info": self.license_info,
            "swagger_ui_parameters": self.swagger_ui_parameters
        }

    def configure_logging(self) -> None:
        logging.getLogger().handlers = [InterceptHandler()]
        for logger_name in self.loggers:
            logging_logger = logging.getLogger(logger_name)
            logging_logger.handlers = [InterceptHandler(level=self.logging_level)]

        logger.configure(handlers=[{"sink": sys.stderr, "level": self.logging_level}])
