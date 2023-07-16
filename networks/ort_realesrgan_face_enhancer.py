import copy
import tempfile
from pathlib import Path
import cv2
from numba import jit, njit
import numpy as np
import onnxruntime
import logging as log
from common.downloer_base import DownloaderBase

# Model from: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/133_Real-ESRGAN
# This is the ONNX model for ORT mainly for aarch64, should also run on Intel CPU
model_name = "realesrgan_128x128"
model_file_name = f'{model_name}.onnx'
model_fp16_file_name = f'{model_name}-fp16.onnx'


class FaceEnhancer:
    def __init__(self, onnx_session, input_shape):
        self.onnx_session = onnx_session
        self.input_shape = input_shape

    def preprocess(self, frame):
        input_image = cv2.resize(frame, dsize=(self.input_shape[1], self.input_shape[0]))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        input_image = input_image / 255.0
        return input_image

    def infer(self, frame):
        preprocessed = self.preprocess(frame)
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        result = self.onnx_session.run([output_name], {input_name: preprocessed})
        postprocessed = self.postprocess(result)
        return postprocessed

    @jit(forceobj=True)
    def postprocess(self, frame):
        outputs = frame
        hr_image = np.squeeze(outputs)
        hr_image = hr_image.transpose(1, 2, 0)
        hr_image = np.clip((hr_image * 255), 0, 255).astype(np.uint8)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR)
        return hr_image


class ORTRealESRGANFaceEnhancer(DownloaderBase):
    def __init__(self, assets_folder: Path):
        super().__init__()
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False
        model_full_path = self.assets_folder / model_file_name
        #self._convert_fp16()
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 4
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        providersList = onnxruntime.capi._pybind_state.get_available_providers()
        onnx_session = onnxruntime.InferenceSession(str(model_full_path), sess_options=options, providers=providersList)
        self.face_enhancer = FaceEnhancer(onnx_session, (128, 128))

    def _convert_fp16(self):
        import onnx
        from onnxconverter_common import float16
        model_full_path = self.assets_folder / model_file_name

        model = onnx.load(model_full_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, self.assets_folder / model_fp16_file_name)

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

        image_enhanced = self.face_enhancer.infer(cropped_img)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file_name = tmp_file.name
        cv2.imwrite(f'{temp_file_name}.png', image_enhanced)  # Background task will delete this file

        return f'{temp_file_name}.png'
