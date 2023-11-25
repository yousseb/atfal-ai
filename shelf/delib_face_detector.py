from pathlib import Path
import dlib
import cv2
from app.common.common_utils import CommonUtilsMixin


class DLibFaceDetector(CommonUtilsMixin):
    def __init__(self, assets_folder: Path):
        self.assets_folder: Path = assets_folder
        self._DEBUG: bool = False
        face_detector_path = str(Path(self.assets_folder / 'mmod_human_face_detector.dat').absolute())
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)

    def detect_faces(self, image_url: str):
        local_image = str(self.download_image(image_url).absolute())

        image = cv2.imread(local_image)
        if self._DEBUG:
            output_image = image.copy()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.cnn_face_detector(img_rgb, upsample_num_times=2)

        boxes = []
        for bbox in results:
            x1_adj, x2_adj, y1_adj, y2_adj = self.adjust_bounding_box(bbox)
            boxes.append({'x1': x1_adj, 'x2': x2_adj, 'y1': y1_adj, 'y2': y2_adj})

            if self._DEBUG:
                cv2.rectangle(output_image, pt1=(x1_adj, y1_adj), pt2=(x2_adj, y2_adj), color=(0, 0, 255), thickness=2)
                cv2.imwrite('test.png', output_image)

        Path(local_image).unlink(missing_ok=True)
        return boxes

    def adjust_bounding_box(self, bbox):
        x1 = bbox.rect.left()
        y1 = bbox.rect.top()
        x2 = bbox.rect.right()
        y2 = bbox.rect.bottom()
        w = x2 - x1
        h = y2 - y1
        x_adj = int(0.5 * w)
        y_adj = int(0.5 * h)
        x1_adj = x1 - x_adj
        y1_adj = y1 - y_adj
        x2_adj = x2 + x_adj
        y2_adj = y2 + y_adj
        if x1_adj < 0:
            x1_adj = 0
        if y1_adj < 0:
            y1_adj = 0
        return x1_adj, x2_adj, y1_adj, y2_adj
