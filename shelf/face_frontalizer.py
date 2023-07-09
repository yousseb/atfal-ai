import tempfile
from pathlib import Path
import cv2
from common.common_utils import CommonUtilsMixin
import torch
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils


class FaceFrontalizer(CommonUtilsMixin):
    def __init__(self, assets_folder: Path):
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False

        import sys
        import os
        sys.path.append(str((Path(os.getcwd()) / 'face_frontalization').absolute()))
        model_path = str((self.assets_folder / 'generator_v0.pt').absolute())
        self.saved_model = torch.load(model_path, map_location=torch.device('cpu'))
        print('Model successfully loaded!')

    def frontaliza_face(self, image_url: str):
        local_image = str(self.download_image(image_url).absolute())
        # img = Image.open(local_image)
        # img.load()
        # image = np.asarray(img)
        image = cv2.imread(local_image, cv2.IMREAD_COLOR)

        preprocess = transforms.Compose((transforms.ToPILImage(),
                                         transforms.Resize(size=(128, 128)),
                                         transforms.ToTensor()))
        input_tensor = torch.unsqueeze(preprocess(image), 0)

        # Use the saved model to generate an output (whose values go between -1 and 1,
        # and this will need to get fixed before the output is displayed)
        generated_image = self.saved_model(Variable(input_tensor.type('torch.FloatTensor')))
        # generated_image = generated_image.detach().squeeze().permute(1, 2, 0).numpy()
        # generated_image = (generated_image + 1.0) / 2.0

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        #cv2.imwrite(f'{temp_file_name}.png', generated_image)          # Background task will delete this file
        #im = Image.fromarray(generated_image)
        #im.save(f'{temp_file_name}.png')
        vutils.save_image(generated_image, f'{temp_file_name}.png', normalize=True)

        print(f'{temp_file_name}.png')
        return f'{temp_file_name}.png'


# Test
if __name__ == '__main__':
    import os
    ASSETS_FOLDER = Path(os.getcwd()) / 'assets'
    FaceFrontalizer(ASSETS_FOLDER).align_faces()
