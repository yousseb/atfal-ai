# Based on https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/face_recognition_demo/python/face_detector.py
# TODO: Compare performance if we use OpenVino PrePostProcessor
#       https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Preprocessing_Details.html
#       Or use with OpenCV dnn
#       https://inspirisys.com/public/blog-details/Low-Code-Face-Detection-with-OpenVINO-toolkit/154

from pathlib import Path
import cv2
from numpy._typing import NDArray
from numba import float32  # import the types
from api.models import Box
from app.common.common_utils import CommonUtilsMixin, resize_input
import numpy as np
from openvino.runtime import PartialShape, get_version
from common.ie_common import Module, OutputTransform
import logging as log

model_name = "face-detection-adas-0001"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'


class FaceDetector(Module):
    class Result(object):
        def __init__(self, output):
            self.image_id: int = output[0]
            self.label: int = int(output[1])
            self.confidence: float = output[2]
            self.position: NDArray[float32] = np.array((output[3], output[4]))  # (x, y)
            self.size: NDArray[float32] = np.array((output[5], output[6]))  # (w, h)

        def rescale_roi(self, roi_scale_factor: float = 1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width: int, frame_height: int):
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):
            min = np.array([0, 0])
            max = np.array([width, height])
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)

    def __init__(self, core, model, input_size, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(core, model, 'Face Detection')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape({self.input_tensor_name: PartialShape([1, 3, *input_size])})
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError("Both input height and width should be positive for Face Detector reshape")

        self.input_shape = list(self.model.inputs[0].shape)
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape
        if len(self.output_shape) != 4 or self.output_shape[3] != 7:
            raise RuntimeError("The model expects output shape with 7 outputs")

        if confidence_threshold > 1.0 or confidence_threshold < 0:
            raise ValueError("Confidence threshold is expected to be in range [0; 1]")
        if roi_scale_factor < 0.0:
            raise ValueError("Expected positive ROI scale factor")

        self.confidence_threshold = confidence_threshold
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        self.input_size = frame.shape
        return resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(FaceDetector, self).enqueue({self.input_tensor_name: input})

    async def postprocess(self):
        outputs = (await self.get_outputs())[0]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break  # results are sorted by confidence decrease

            result.resize_roi(self.input_size[1], self.input_size[0])
            result.rescale_roi(self.roi_scale_factor)
            result.clip(self.input_size[1], self.input_size[0])
            results.append(result)

        return results


class OVFaceDetector(CommonUtilsMixin):
    def __init__(self, assets_folder: Path):
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False
        core = self.get_core()
        model_xml_path = self.assets_folder / model_xml_name
        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))

        self.face_detector = FaceDetector(core, model_xml_path,
                                          (0, 0),
                                          confidence_threshold=0.6,
                                          roi_scale_factor=1.15)
        self.face_detector.deploy('CPU')

    async def detect_faces(self, image_url: str) -> list[Box]:
        local_image = str(self.download_image(image_url).absolute())
        image = cv2.imread(local_image)
        if self._DEBUG:
            output_image = image.copy()
        Path(local_image).unlink(missing_ok=True)

        size = image.shape[:2]
        output_transform = OutputTransform(size, None)

        # numpy_image = np.asarray(image)
        rois = await self.face_detector.infer((image,))
        boxes = []
        await self.rois_to_boxes(boxes, output_transform, rois, size)

        # if self._DEBUG:
        #     cv2.rectangle(output_image, pt1=(box.x1, box.y1), pt2=(box.x2, box.y2), color=(0, 0, 255),
        #                   thickness=2)
        #     cv2.imwrite('test.png', output_image)

        return boxes

    async def rois_to_boxes(self, boxes, output_transform, rois, size):
        for roi in rois:
            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), size[1])
            ymax = min(int(roi.position[1] + roi.size[1]), size[0])
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])

            box = Box(x1=xmin, x2=xmax, y1=ymin, y2=ymax)
            boxes.append(box)
