import tempfile
from pathlib import Path
import cv2
from common.common_utils import CommonUtilsMixin
import torch.backends.cudnn as cudnn
import torch
import sys
import os


class CFRGANFaceFrontalizer(CommonUtilsMixin):
    def __init__(self, assets_folder: Path):
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False

        estimator_model_path = str(Path(self.assets_folder / 'trained_weights_occ_3d.pth').absolute())
        cudnn.benchmark = True
        det_net = None
        sys.path.append(str((Path(os.getcwd()) / 'networks' / 'cfrgan').absolute()))
        from generate_pairs import Estimator3D
        from model.networks import CFRNet

        self.estimator3d = Estimator3D(is_cuda=False, render_size=224, assets_path=self.assets_folder,
                                       model_path=estimator_model_path, det_net=det_net,
                                       cuda_id=-1)
        self.cfrnet = CFRNet().cpu()
        generator_path = str(Path(self.assets_folder / 'CFRNet_G_ep55_vgg.pth').absolute())
        trained_weights = torch.load(generator_path, map_location=torch.device('cpu'))
        own_state = self.cfrnet.state_dict()

        for name, param in trained_weights.items():
            own_state[name[7:]].copy_(param)
        self.cfrnet.eval()

        # self.saved_model = torch.load(model_path, map_location=torch.device('cpu'))
        print('Model successfully loaded!')

    # def check_keys(self, model, pretrained_state_dict):
    #     ckpt_keys = set(pretrained_state_dict.keys())
    #     model_keys = set(model.state_dict().keys())
    #     used_pretrained_keys = model_keys & ckpt_keys
    #     unused_pretrained_keys = ckpt_keys - model_keys
    #     missing_keys = model_keys - ckpt_keys
    #     print('Missing keys:{}'.format(len(missing_keys)))
    #     print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #     print('Used keys:{}'.format(len(used_pretrained_keys)))
    #     assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    #     return True
    #
    # def remove_prefix(self, state_dict, prefix):
    #     ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #     print('remove prefix \'{}\''.format(prefix))
    #     f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    #     return {f(key): value for key, value in state_dict.items()}
    #
    # def load_model(self, model, pretrained_path, load_to_cpu):
    #     print('Loading pretrained model from {}'.format(pretrained_path))
    #     if load_to_cpu:
    #         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    #     else:
    #         device = torch.cuda.current_device()
    #         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    #     if "state_dict" in pretrained_dict.keys():
    #         pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
    #     else:
    #         pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
    #     self.check_keys(model, pretrained_dict)
    #     model.load_state_dict(pretrained_dict, strict=False)
    #     return model

    def normalize(self, img):
        return (img - 0.5) * 2

    def frontaliza_face(self, image_url: str):
        local_image = str(self.download_image(image_url).absolute())
        image_list = [local_image]

        input_img = self.estimator3d.align_convert2tensor(image_list, aligned=True)

        # rotated: RGB, guidance: BGR
        rotated, guidance = self.estimator3d.generate_testing_pairs(input_img, pose=[5.0, 0.0, 0.0])

        rotated = self.normalize(rotated[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous())
        guidance = self.normalize(guidance[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous())
        output, occ_mask = self.cfrnet(rotated, guidance)
        output = (output / 2) + 0.5
        output = (output.permute(0, 2, 3, 1) * 255).cpu().detach().numpy().astype('uint8')

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name

        for i in range(rotated.shape[0]):
            cv2.imwrite(f'{temp_file_name}.png',
                        cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))

        print(f'{temp_file_name}.png')
        return f'{temp_file_name}.png'


# Test
# if __name__ == '__main__':
#     import os
#
#     ASSETS_FOLDER = Path(os.getcwd()) / 'assets'
#     FaceFrontalizer(ASSETS_FOLDER).align_faces()
