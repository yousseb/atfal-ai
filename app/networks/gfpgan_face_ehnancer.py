import copy
import tempfile
from pathlib import Path

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
import cv2

from app.common.downloer_base import DownloaderBase

upsampler_model_name = 'realesr-general-x4v3.pth'
gfpgan_model_name = 'RestoreFormer.pth'


class GFPGANFaceEnhancer(DownloaderBase):
    def __init__(self, assets_folder: Path, debug: bool = False):
        super().__init__()
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = debug
        # background enhancer with RealESRGAN
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        upsampler_model_path = str((self.assets_folder / upsampler_model_name).absolute())
        gfpgan_model_path = str((self.assets_folder / gfpgan_model_name).absolute())
        half = False
        upsampler = RealESRGANer(scale=4, model_path=upsampler_model_path, model=model, tile=0, tile_pad=10, pre_pad=0,
                                 half=half)
        self.face_enhancer = GFPGANer(
            model_path=gfpgan_model_path,
            upscale=2, arch='RestoreFormer',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # def inference(img, version, scale, weight):
    def infer(self, img, scale):
        if len(img.shape) == 3 and img.shape[2] == 4:
            pass
        elif len(img.shape) == 2:  # for gray inputs
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        # _, _, output = face_enhancer.enhance(img, has_aligned=False,
        # only_center_face=False, paste_back=True, weight=weight)
        _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False,
                                                  paste_back=True)

        if scale != 2:
            interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
            h, w = img.shape[0:2]
            output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        return output

    def get_best_input_rect(self, box, image_shape):
        new_box = copy.deepcopy(box)
        image_height, image_width = image_shape

        for i in range(5):  # A recursive function would be better. But could be an overkill, too.
            w = new_box.x2 - new_box.x1
            h = new_box.y2 - new_box.y1

            if w == h:
                return new_box

            if w > h:
                diff = w - h
                margin = int(diff / 2)
                new_box.y1 = max(new_box.y1 - margin, 0)
                new_box.y2 = min(new_box.y2 + margin, image_height)
            elif w < h:
                diff = h - w
                margin = int(diff / 2)
                new_box.x1 = max(new_box.x1 - margin, 0)
                new_box.x2 = min(new_box.x2 + margin, image_width)

        return new_box

    def enhance_face(self, image_url: str, box):
        local_image = str(self.download_image(image_url).absolute())
        full_image = cv2.imread(local_image, cv2.IMREAD_UNCHANGED)
        Path(local_image).unlink(missing_ok=True)

        new_box = self.get_best_input_rect(box, full_image.shape[:2])

        cropped_img = full_image[new_box.y1:new_box.y2, new_box.x1:new_box.x2]
        if self._DEBUG:
            input_image = cropped_img.copy()

        image_enhanced = self.infer(cropped_img, 2)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        cv2.imwrite(f'{temp_file_name}.png', image_enhanced)  # Background task will delete this file

        return f'{temp_file_name}.png'
