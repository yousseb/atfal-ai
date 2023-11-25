# TODO: Check if we can user this instead: https://github.com/scaleway/frontalization
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import sys
import os
from app.common.downloer_base import DownloaderBase
from .cfrgan.generate_pairs import Estimator3D
from .cfrgan.model.networks import CFRNet


class CFRGANFaceFrontalizer(DownloaderBase):
    def __init__(self, assets_folder: Path):
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False

        estimator_model_path = str(Path(self.assets_folder / 'trained_weights_occ_3d.pth').absolute())
        cudnn.benchmark = True
        det_net = None

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

        #self._cfrnet_export_to_onnx(rotated, guidance)

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

    def _cfrnet_export_to_onnx(self, rotated, guidance):
        sys.path.append(str((Path(os.getcwd()) / 'networks' / 'cfrgan').absolute()))
        from model.networks import CFRNet
        import onnx
        import onnxruntime

        onnx_model_path = str(Path(self.assets_folder / 'CFRNet_G_ep55_vgg.onnx').absolute())

        torch.save(self.cfrnet.state_dict(), str(Path(self.assets_folder / 'CFRNet_G_ep55_vgg_saved.pth').absolute()))
        self.cfrnet = CFRNet().cpu()
        self.cfrnet.load_state_dict(torch.load(str(Path(self.assets_folder / 'CFRNet_G_ep55_vgg_saved.pth').absolute())))
        self.cfrnet.eval()

        torch.onnx.export(self.cfrnet,
                          args=(rotated, guidance),
                          f=onnx_model_path,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=['rotated', 'guidance'],
                          output_names=['output', 'occ_mask'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}}
                          )

        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy()

        output, occ_mask = self.cfrnet(rotated, guidance)

        # compute ONNX Runtime output prediction
        ort_output, ort_occ_mask = ort_session.run(None, {
            ort_session.get_inputs()[0].name: rotated.cpu().detach().numpy(),
            ort_session.get_inputs()[1].name: guidance.cpu().detach().numpy()
        })

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(output), ort_output, rtol=1e-01, atol=1e-07)
        print('output match...')

        np.testing.assert_allclose(to_numpy(occ_mask), ort_occ_mask, rtol=1e-01, atol=1e-07)
        print('occ_mask match...')
