import tempfile
from pathlib import Path
import cv2
from app.common.common_utils import CommonUtilsMixin
import torch
import shelf.ESRGAN.RRDBNet_arch as arch
import numpy as np

# TODO: Replace with https://github.com/cszn/BSRGAN


class RRDBESRGANFaceEnhancer(CommonUtilsMixin):
    def __init__(self, assets_folder: Path):
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False

        model_path = self.assets_folder / 'RRDB_ESRGAN_x4.pth'  # load the model
        self.device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
        # device = torch.device('cuda')
        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(str(model_path.absolute())), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def enhance_face(self, image_url: str, box):
        local_image = str(self.download_image(image_url).absolute())
        img = cv2.imread(local_image, cv2.IMREAD_COLOR)
        Path(local_image).unlink(missing_ok=True)

        img = img[box.y1:box.y2, box.x1:box.x2]

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_low_res = img.unsqueeze(0)
        img_low_res = img_low_res.to(self.device)

        with torch.no_grad():
            image_enhanced = self.model(img_low_res).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        image_enhanced = np.transpose(image_enhanced[[2, 1, 0], :, :], (1, 2, 0))
        image_enhanced = (image_enhanced * 255.0).round()

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        cv2.imwrite(f'{temp_file_name}.png', image_enhanced)          # Background task will delete this file

        return f'{temp_file_name}.png'
