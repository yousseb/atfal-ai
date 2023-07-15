# Mostly based on https://github.com/discipleofhamilton/RetinaFace/tree/master
from pathlib import Path

import cv2
from api.models import Box
from common.downloer_base import DownloaderBase
from networks.retinaface.retinaface import RetinaFaceDetector

model_name = 'retinaface_640x640_opt.onnx'
prior_boxes = 'priorbox_640x640.json'


class ORTFaceDetector(DownloaderBase):
    def __init__(self, assets_folder: Path):
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False
        model_path = self.assets_folder/model_name
        json_priors_path = self.assets_folder/prior_boxes
        self.face_detector = RetinaFaceDetector(model_path=str(model_path.absolute()),
                                                json_path=str(json_priors_path.absolute()),
                                                top_k=40, min_conf=0.2)

    def detect_faces(self, image_url: str):
        local_image = str(self.download_image(image_url).absolute())
        image = cv2.imread(local_image, cv2.IMREAD_COLOR)
        if self._DEBUG:
            output_image = image.copy()
        Path(local_image).unlink(missing_ok=True)

        faces = self.face_detector.detect_retina(image)
        boxes = []
        for (x, y, w, h) in faces:
            box = Box(x1=x, x2=x+w, y1=y, y2=y+h)
            boxes.append(box)

        if self._DEBUG:
            for box in boxes:
                cv2.rectangle(output_image, pt1=(box.x1, box.y1), pt2=(box.x2, box.y2), color=(0, 0, 255),
                              thickness=2)
                cv2.imwrite(str((self.assets_folder/'../test.jpg').absolute()), output_image)

        return boxes
