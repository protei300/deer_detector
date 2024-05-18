import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Union
import cv2


class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class BaseModel:


    def __init__(self, model_path: Union[Path, str], **kwargs):
        self.model_path = model_path

    def predict(self, img: np.ndarray, **kwargs):
        pass

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

class YOLOModel(BaseModel):
    """ Any ONNX Model """
    def __init__(self, model_path: Union[Path,str], **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = ort.InferenceSession(self.model_path)
        inp_shape = self.model.get_inputs()[0].shape[1:]
        self.inp_shape = inp_shape
        ort_inputs = {self.model.get_inputs()[0].name: np.random.rand(*np.append(1, inp_shape)).astype(np.float32)}
        self.model.run(None, ort_inputs)
        self.labels = kwargs.get('labels')
        self.iou = kwargs.get('iou', 0.2)
        self.conf = kwargs.get('conf', 0.5)

        self.colors = Colors()

        print(f"[+] YOLO Model is ready!")

    def predict(self, img_path: Union[np.ndarray, Path, str], **kwargs):
        img = self.open_image(img_path)

        new_shape = self.inp_shape[:2] if self.inp_shape[-1] == 3 else self.inp_shape[1:]
        img, ratio, padding = self.letterbox(img, new_shape=new_shape, auto=False)
        padding_meta = {
            'x_offset': padding[0],
            'y_offset': padding[1],
            'ratio': ratio,
        }

        pred = self.__inference(img)

        # working with 1 picture only
        pred = np.squeeze(pred)

        postprocess_result = self.__postprocess(
            prediction=pred,
            padding_meta=padding_meta,
            labels=self.labels,
            imgsize=img.shape[:2],
            conf_thresh=self.conf,
            iou_thresh=self.iou,
        )

        return postprocess_result


    def predict_draw(self, img_path: Union[np.ndarray, Path, str], **kwargs) -> (list, np.ndarray):
        img = self.open_image(img_path)
        boxes = self.predict(img, **kwargs)
        for bbox in boxes:

            label, label_i, conf = (bbox[-1], bbox[-2], bbox[-3]) if self.labels else (bbox[-1], bbox[-1], bbox[-2])
            color = self.colors(label_i)[::-1]
            img = cv2.rectangle(img, bbox[:2], bbox[2:4], color=color, thickness=2)
            img = cv2.putText(img,
                              f"{label}_{conf:.02f}",
                              bbox[:2],
                              0,
                              1,
                              color,
                              thickness=3,
                              lineType=cv2.LINE_AA)
        return boxes, img

    def __inference(self, tensor):
        input_name = self.model.get_inputs()[0].name
        input_shape = tuple(self.model.get_inputs()[0].shape[1:3])

        if input_shape[-1] != 3:
            tensor = np.transpose(tensor, [2, 0, 1])
        tensor = tensor.astype(np.float32)
        tensor /= 255
        ort_inputs = {input_name: tensor[None, ...]}
        ort_outputs = self.model.run(None, ort_inputs)
        pred = ort_outputs[0] if len(ort_outputs) == 1 else ort_outputs

        return pred



    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        '''
        Function used for letterbox resize. It resize and uses padding to fill picture to new_shape size
        :param im: image
        :param new_shape: shape of new image
        :param color: color for padding fill
        :param auto: Make minimum rectangle as possible
        :param scaleFill: should we use stretch
        :param scaleup: IF needed should we use scaleup
        :param stride: size, of squares. Used for Yolo5, default 32
        :return:  new_image, ratio which been used to reduce size, padding size
        '''
        # Resize and pad image while meeting stride-multiple constraints

        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)



    @staticmethod
    def nms(bounding_boxes: np.array, confidence_score: np.array, threshold: float):
        '''
        Finds best boxes for found objects
        :param bounding_boxes: coords of boxes
        :param confidence_score: np.ndarray with shape (n, 1)
        :param threshold: IoU_threshold
        :return: array of indexes, that corresponds to best found BOXES
        '''

        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        # Picked bounding boxes
        picked_boxes_index = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x) * (end_y - start_y)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score, axis=0).reshape(-1)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes_index.append(index)

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]

        return picked_boxes_index,

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def xyxy2xywh(x):
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # top left x
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # top left y
        y[:, 2] = x[:, 2] - x[:, 0]  # bottom right x
        y[:, 3] = x[:, 3] - x[:, 1]  # bottom right y
        return y

    def __postprocess(
            self,
            prediction: np.array,
            imgsize=(640, 640),
            conf_thresh=0.5,
            iou_thresh=0.45,
            padding_meta=None,
            labels=None
    ) -> Union[np.ndarray, list]:
        '''
        Function translates predictions from yolo detector
        :param prediction: Tensor from yolo exit
        :param imgsize: size of image, h,w
        :param conf_thresh: class confidence threshold
        :param iou_thresh:  intersection over union threshold
        :return:
        '''

        if padding_meta is None:
            padding_meta = {
                'x_offset': 0,
                'y_offset': 0,
                'ratio': (1, 1),
            }

        y, x = imgsize

        if prediction.shape[-1] > prediction.shape[-2]:
            prediction = np.transpose(prediction, [1, 0])

        confidence = prediction[..., 4:].max(axis=1) > conf_thresh
        prediction = prediction[confidence]

        detect_res = []

        if prediction.shape[0] == 0:
            return detect_res


        box = self.xywh2xyxy(prediction[..., :4])  # creating from x,y height, width -> xy xy coords of box
        conf, j = prediction[:, 4:].max(axis=1, keepdims=True), prediction[:, 4:].argmax(axis=1, keepdims=True)

        i = self.nms(box, conf, iou_thresh)  # calculating non maximum suppression

        detect_res = np.concatenate((box, conf, j), axis=1)[i]

        ### resize to original pic

        detect_res[..., :4:2] = (detect_res[..., :4:2] - padding_meta['x_offset']) / padding_meta['ratio'][0]
        detect_res[..., 1:4:2] = (detect_res[..., 1:4:2] - padding_meta['y_offset']) / padding_meta['ratio'][1]
        detect_res[detect_res < 0] = 0

        detect_res[..., :4] = np.round(detect_res[:, :4], 0)
        detect_res[..., 4] = np.round(detect_res[:, 4], 3)

        #sorting
        ordered_index = np.lexsort((detect_res[...,1], detect_res[..., 0]), axis=0)
        detect_res = detect_res[ordered_index]


        detection_result = []
        for detected in detect_res:
            obj_detected = list(detected)
            obj_detected[:4] = list(map(lambda x: int(x), obj_detected[:4]))
            obj_detected[5] = int(obj_detected[5])
            if labels is not None:
                try:
                    obj_detected.append(labels[obj_detected[5]])
                except:
                    obj_detected.append("Unsupported class")

            detection_result.append(obj_detected)
        return detection_result
        # else:
        #
        #     obj_detected[:4] = list(map(lambda x: int(x), obj_detected[:4]))
        #     return detect_res.tolist()





class ONNXClassification(BaseModel):
    """ ONNX MODEL Binary"""
    def __init__(self, model_path: Path, **kwargs):
        """Classification model for ONNX"""

        super().__init__(model_path, **kwargs)


        labels_file = model_path.with_suffix('.txt')

        assert labels_file.exists(), "[!] No labels file specified"
        self.labels = labels_file.read_text().splitlines()
        self.model = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider', ])
        inp_shape = self.model.get_inputs()[0].shape[1:]
        self.inp_shape = inp_shape
        self.imgsz = self.inp_shape[1:] if self.inp_shape[0] == 3 else self.inp_shape[:2]
        ort_inputs = {self.model.get_inputs()[0].name: np.random.rand(*np.append(1, inp_shape)).astype(np.float32)}
        self.model.run(None, ort_inputs)
        print(f"[+] Classification Model is ready!")

    def predict(self, img_path: Union[np.ndarray, Path, str], **kwargs):
        img = self.open_image(img_path)
        img = self.__preprocess(img)
        pred = self.__inference(img)
        pred = np.squeeze(pred)
        postprocess_result = self.__postprocess(prediction=pred)
        return postprocess_result


    def __inference(self, tensor):
        input_name = self.model.get_inputs()[0].name
        input_shape = tuple(self.model.get_inputs()[0].shape[1:4])
        if input_shape[0] == 3:
            tensor = np.transpose(tensor, [2,0,1])
        tensor = tensor.astype(np.float32)
        ort_inputs = {input_name: tensor[None, ...]}
        ort_outputs = self.model.run(None, ort_inputs)
        pred = ort_outputs[0] if len(ort_outputs) == 1 else ort_outputs

        return pred

    def __preprocess(self, img):
        img = cv2.resize(img, self.imgsz, cv2.INTER_LINEAR)
        return img

    def __postprocess(self, prediction):
        label = self.labels[np.argmax(prediction)]
        return label
