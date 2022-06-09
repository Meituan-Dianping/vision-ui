import os.path
import re
import cv2
import numpy as np
import onnxruntime
import time
from config import IMAGE_INFER_MODEL_PATH, OP_NUM_THREADS
from service.image_utils import yolox_preprocess, yolox_postprocess, multiclass_nms, img_show


class ImageInfer(object):
    def __init__(self, model_path):
        self.UI_CLASSES = ("bg", "icon", "pic")
        self.input_shape = [640, 640]
        self.cls_thresh = 0.5
        self.nms_thresh = 0.2
        self.model_path = model_path
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = OP_NUM_THREADS
        self.model_session = onnxruntime.InferenceSession(self.model_path, sess_options=so)

    def ui_infer(self, image_path):
        origin_img = cv2.imread(image_path)
        img, ratio = yolox_preprocess(origin_img, self.input_shape)
        ort_inputs = {self.model_session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.model_session.run(None, ort_inputs)
        predictions = yolox_postprocess(output[0], self.input_shape)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nms_thresh, score_thr=self.cls_thresh)
        if dets is not None:
            # 兼容不同版本模型返回结果中UI classes index起始位置
            offset = 0
            match_obj = re.match(r'.*o(\d+)\.onnx$', self.model_path)
            if match_obj:
                offset = int(match_obj.group(1))
            dets[:, 5] += offset
        return dets

    def show_infer(self, dets, origin_img, infer_result_path):
        if dets is not None:
            boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = img_show(origin_img, boxes, scores, cls_inds, conf=self.cls_thresh,
                                  class_names=self.UI_CLASSES)
        cv2.imwrite(infer_result_path, origin_img)


image_infer = ImageInfer(IMAGE_INFER_MODEL_PATH)


def get_ui_infer(image_path, cls_thresh):
    """
    elem_det_region x1,y1,x2,y2
    """
    data = []
    image_infer.cls_thresh = cls_thresh if isinstance(cls_thresh, float) else image_infer.cls_thresh
    dets = image_infer.ui_infer(image_path)
    if isinstance(dets, np.ndarray):
        boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for i in range(len(boxes)):
            box = boxes[i]
            box[box < 0] = 0
            box = box.tolist() if isinstance(box, (np.ndarray,)) else box
            elem_type = image_infer.UI_CLASSES[int(cls_inds[i])]
            score = scores[i]
            data.append(
                {
                    "elem_det_type": "image" if elem_type == 'pic' else elem_type,
                    "elem_det_region": box,
                    "probability": score
                }
            )
    return data


if __name__ == '__main__':
    """
    调试代码
    """
    image_path = "./capture/image_1.png"
    infer_result_path = "./capture/local_images"
    assert os.path.exists(image_path)
    assert os.path.exists(IMAGE_INFER_MODEL_PATH)
    if not os.path.exists(infer_result_path):
        os.mkdir(infer_result_path)
    t1 = time.time()
    dets = image_infer.ui_infer(image_path)
    print(f"Infer time: {round(time.time()-t1, 3)}s")
    infer_result_name = f"infer_{str(time.time()).split('.')[-1][:4]}.png"
    image_infer.show_infer(dets, cv2.imread(image_path), os.path.join(infer_result_path, infer_result_name))
    print(f"Result saved {infer_result_path}/{infer_result_name}")
