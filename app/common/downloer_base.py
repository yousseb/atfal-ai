import tempfile
from pathlib import Path
from typing import Optional

import hishel
import httpx
from loguru import logger


class DownloaderBase:
    http_client: Optional[httpx.Client] = None

    def __init__(self):
        pass

    @classmethod
    def get_http_client(cls) -> httpx.Client:
        if cls.http_client is None:
            limits = httpx.Limits(
                max_connections=50, max_keepalive_connections=10, keepalive_expiry=5.0
            )
            timeout = httpx.Timeout(20.0, connect=60.0)
            storage = hishel.FileStorage(
                ttl=600, base_path=Path(tempfile.gettempdir())
            )
            transport = httpx.HTTPTransport(http2=True, limits=limits, retries=3)
            controller = hishel.Controller(
                cacheable_methods=["GET"], cacheable_status_codes=[200]
            )
            cache_transport = hishel.CacheTransport(
                storage=storage, transport=transport, controller=controller
            )
            cls.http_client = httpx.Client(
                transport=cache_transport, timeout=timeout, limits=limits, follow_redirects=True
            )
        return cls.http_client

    @classmethod
    def download_file(cls, url: str, path: Path):
        client: httpx.Client = cls.get_http_client()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.35 (KHTML, like Gecko) '
                          'Chrome/39.0.2171.95 Safari/537.36'}
        with httpx.stream("GET", url, headers=headers, follow_redirects=True) as response:
            if response.status_code != 200:
                logger.error(f"Error downloading file. Status code: {response.status_code}")
                return False
            with open(path, 'wb') as f:
                for chunk in response.iter_bytes():
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
        return True

    @classmethod
    def download_image(cls, image_url: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        cls.download_file(image_url, Path(temp_file_name))
        return Path(temp_file_name)
