import tempfile
from pathlib import Path
import cv2
import requests
from openvino.runtime import Core
from numba import jit


@jit(forceobj=True)
def resize_image(image, size, keep_aspect_ratio=False, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size, interpolation=interpolation)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized_frame


@jit(forceobj=True)
def resize_input(image, target_shape, nchw_layout):
    if nchw_layout:
        _, _, h, w = target_shape
    else:
        _, h, w, _ = target_shape
    resized_image = resize_image(image, (w, h))
    if nchw_layout:
        resized_image = resized_image.transpose((2, 0, 1))  # HWC->CHW
    resized_image = resized_image.reshape(target_shape)
    return resized_image


class CoreManager(object):
    _core = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CoreManager, cls).__new__(cls)
        return cls.instance

    def get_core(self):
        if self._core is None:
            self._core = Core()
            return self._core
        else:
            return self._core


class CommonUtilsMixin:
    _core_manager = CoreManager()

    def download_file(self, url: str, path: Path):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.35 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = requests.get(url, headers=headers)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        return

    def download_image(self, image_url: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        self.download_file(image_url, Path(temp_file_name))
        return Path(temp_file_name)

    def get_core(self):
        return self._core_manager.get_core()
