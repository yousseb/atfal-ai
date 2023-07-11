import cv2
from openvino.runtime import Core
from openvino.inference_engine import IECore
from numba import jit
from common.downloer_base import DownloaderBase


@jit(forceobj=True)
def resize_image(image, size, keep_aspect_ratio=False, interpolation=cv2.INTER_CUBIC):
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
            ie = IECore()
            cpu_caps = ie.get_metric(metric_name="OPTIMIZATION_CAPABILITIES", device_name="CPU")
            print(f'OpenVino Available CPU Optimizations: {cpu_caps}')
            self._core = Core()
            self._core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "f32"})

            if 'BF16' in cpu_caps:
                self._core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "bf16"})
                self._core.set_property({'ENFORCE_BF16': 'YES'})
            return self._core
        else:
            return self._core


class CommonUtilsMixin(DownloaderBase):
    _core_manager = CoreManager()

    def get_core(self):
        return self._core_manager.get_core()
