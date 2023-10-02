from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from starlette.status import HTTP_403_FORBIDDEN
from functools import lru_cache
from config import Settings

api_key_header = APIKeyHeader(name="access_token", auto_error=False)


@lru_cache()
def get_settings():
    return Settings()


async def api_key_auth(header: str = Security(api_key_header)):
    if header in get_settings().API_KEYS:
        return header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY"
        )
