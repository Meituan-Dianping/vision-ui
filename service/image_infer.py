import os.path
import cv2
import numpy as np
import onnxruntime
import time
from service.image_utils import yolox_preprocess, yolox_postprocess, multiclass_nms, img_show


class ImageInfer(object):
    def __init__(self, model_path):
        self.UI_CLASSES = ("bg", "icon", "pic")
        self.input_shape = [640, 640]
        self.cls_thresh = 0.5
        self.nms_thresh = 0.2
        self.model_path = model_path
        self.model_session = onnxruntime.InferenceSession(self.model_path)

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
        return dets

    def show_infer(self, dets, origin_img, infer_result_path):
        if dets is not None:
            boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = img_show(origin_img, boxes, scores, cls_inds, conf=self.cls_thresh,
                                  class_names=self.UI_CLASSES)
        cv2.imwrite(infer_result_path, origin_img)


if __name__ == '__main__':
    """
    调试代码
    """
    image_path = "../capture/local_images/01.png"
    model_path = "../capture/local_models/ui_det_v1.onnx"
    infer_result_path = "../capture/local_images"
    assert os.path.exists(image_path)
    assert os.path.exists(model_path)
    if not os.path.exists(infer_result_path):
        os.mkdir(infer_result_path)
    image_infer = ImageInfer(model_path)
    t1 = time.time()
    dets = image_infer.ui_infer(image_path)
    print(f"Infer time: {round(time.time()-t1, 3)}s")
    infer_result_name = f"infer_{str(time.time()).split('.')[-1][:4]}.png"
    image_infer.show_infer(dets, cv2.imread(image_path), os.path.join(infer_result_path, infer_result_name))
