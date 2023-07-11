import copy
import tempfile
from pathlib import Path
import cv2
from numba import jit

from common.common_utils import CommonUtilsMixin, resize_input
import numpy as np
from common.ie_common import Module
from openvino.runtime import PartialShape, get_version
import logging as log


# Model from: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/133_Real-ESRGAN
# This is the ONNX model for OpenVino - crashes on aarch64, so this is for Intel CPU/GPU only
model_name = "realesrgan_128x128"
model_xml_name = f'{model_name}.onnx'


class FaceEnhancer(Module):
    def __init__(self, core, model, input_size):
        super(FaceEnhancer, self).__init__(core, model, 'Face Enhancer')
        # from openvino.runtime import serialize
        # serialize(self.model, model_xml_name)

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape({self.input_tensor_name: PartialShape([1, 3, *input_size])})
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError("Both input height and width should be positive for Face Enhancer reshape")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape
        if len(self.output_shape) != 4:
            raise RuntimeError("The model expects output shape with 4 outputs")

    def preprocess(self, frame):
        self.input_size = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        return resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(FaceEnhancer, self).enqueue({self.input_tensor_name: input})

    @jit(forceobj=True)
    def postprocess(self):
        outputs = self.get_outputs()[0]
        hr_image = np.squeeze(outputs)
        hr_image = hr_image.transpose(1, 2, 0)
        hr_image = np.clip((hr_image * 255), 0, 255).astype(np.uint8)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR)
        return hr_image


class OVRealESRGANFaceEnhancer(CommonUtilsMixin):
    def __init__(self, assets_folder: Path):
        super(OVRealESRGANFaceEnhancer, self).__init__()

        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False

        core = self.get_core()
        model_xml_path = self.assets_folder / model_xml_name
        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))

        self.face_enhancer = FaceEnhancer(core, model_xml_path, (0, 0))
        self.face_enhancer.deploy('CPU')

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
        full_image = cv2.imread(local_image, cv2.IMREAD_COLOR)
        Path(local_image).unlink(missing_ok=True)

        new_box = self.get_best_input_rect(box, full_image.shape[:2])

        cropped_img = full_image[new_box.y1:new_box.y2, new_box.x1:new_box.x2]
        if self._DEBUG:
            input_image = cropped_img.copy()

        image_enhanced = self.face_enhancer.infer((cropped_img,))

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        cv2.imwrite(f'{temp_file_name}.png', image_enhanced)  # Background task will delete this file

        return f'{temp_file_name}.png'
