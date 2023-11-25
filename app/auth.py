from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from starlette import status
from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def api_key_auth(header: str = Security(api_key_header)):
    api_keys = get_settings().API_KEYS
    print(f'header: {header}')
    print(f'API_KEYS: {api_keys}')
    if header in api_keys:
        return header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key."
        )
