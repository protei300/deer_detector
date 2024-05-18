from .onnx_processing import YOLOModel, ONNXClassification
from pathlib import Path
import cv2
from typing import Union
import numpy as np



ROOT_FOLDER = Path(__file__).parents[1]


class Pipeline():

    def __init__(self):

        detector_path = ROOT_FOLDER.joinpath('Models', 'Detectors', 'model.onnx')
        self.detector = YOLOModel(detector_path, iou=0.1, conf=0.3, labels=['Deer',],)

        classifier_path = ROOT_FOLDER.joinpath('Models', 'Classification')
        self.classifier = ONNXClassification(classifier_path / 'model_3classes.onnx')
        # self.classifier_dino = ONNXClassification(Path(r'C:\PyProjects\Hackatons\hacks-ai-urfo\Valuev\DeerClassification3\Models\ONNX\model.onnx'))


    def process(self, img_path: Union[Path, str, np.ndarray]):

        self.result = {}

        img = self.open_image(img_path)
        patches = self._detect(img)
        self.result['detector'] = patches

        #### Классификация ######
        self.result['classification'] = self._classify(patches['patches'])
        return self.result['classification']



    def _detect(self, img: list) -> dict:
        boxes, img_w_boxes = self.detector.predict_draw(img)
        patches = []
        for row in boxes:
            x_u, y_u, x_b, y_b = row[:4]
            patch = img[y_u: y_b, x_u: x_b]
            patches.append(patch)
        result = {
            'patches': patches,
            'boxes': boxes,
            'img_w_boxes': img_w_boxes
        }

        return result


    def _classify(self, patches: np.ndarray) -> str:

        clsf_res = {lbl: 0 for lbl in self.classifier.labels}
        # print(clsf_res)
        for patch in patches:
            result = self.classifier.predict(patch)
            # print(result)
            if result in clsf_res.keys():
                clsf_res[result] += 1
        clsf_res = {k: v for k, v in sorted(clsf_res.items(), key=lambda item: item[1], reverse=True)}

        lbls, values = zip(*clsf_res.items())

        if values[0] != 0:
            return lbls[0]
        else:
            return 'NO_DEER'





    @staticmethod
    def open_image(img_path: Union[Path, str, np.ndarray]):

        if isinstance(img_path, Path):
            img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, str):
            img = cv2.imdecode(np.frombuffer(Path(img_path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_path
        return img
