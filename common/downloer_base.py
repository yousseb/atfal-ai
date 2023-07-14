import tempfile
from datetime import timedelta
from pathlib import Path
import requests
from requests_cache import CachedSession


class DownloaderBase:
    def __init__(self):
        pass

    def download_file(self, url: str, path: Path):
        session = CachedSession('http_cache', backend='filesystem', use_temp=True, expire_after=timedelta(minutes=15))

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.35 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = session.get(url, headers=headers)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        return True

    def download_image(self, image_url: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        self.download_file(image_url, Path(temp_file_name))
        return Path(temp_file_name)
