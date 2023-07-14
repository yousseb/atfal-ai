import cv2
from common.downloer_base import DownloaderBase


def resize_image(image, size, keep_aspect_ratio=False, interpolation=cv2.INTER_CUBIC):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size, interpolation=interpolation)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized_frame


def resize_input(image, target_shape, nchw_layout: bool):
    if nchw_layout:
        _, _, h, w = target_shape
    else:
        _, h, w, _ = target_shape
    resized_image = resize_image(image, (w, h))
    if nchw_layout:
        resized_image = resized_image.transpose((2, 0, 1))  # HWC->CHW
    resized_image = resized_image.reshape(target_shape)
    return resized_image


