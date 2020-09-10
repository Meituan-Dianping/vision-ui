from dbnet_crnn_lstm.crnn.crnn import CRNNHandle
from dbnet_crnn_lstm.ocr_utils import sort_boxes, get_rotate_crop_image
from service.image_utils import get_center_pos
from PIL import Image
import numpy as np
import copy
from dbnet_crnn_lstm.dbnet.dbnet_infer import DBNET


class ImageOcr(object):
    def __init__(self):
        self.text_handle = DBNET('dbnet_crnn_lstm/models/dbnet.onnx')
        self.crnn_handle = CRNNHandle('dbnet_crnn_lstm/models/crnn_lstm.onnx')

    def crnn_rect_with_box(self, im, boxes_list, score_list):
        results = []
        boxes_list = sort_boxes(np.array(boxes_list))
        for index, (box, score) in enumerate(zip(boxes_list, score_list)):
            tmp_box = copy.deepcopy(box)
            pos = get_center_pos(tmp_box)
            roi_list = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            roi = Image.fromarray(roi_list).convert("RGB")
            roi = roi.convert('L')
            prediction = self.crnn_handle.predict(roi)
            if prediction.strip() != '':
                results.append({
                    'pos': [pos[0], pos[1]],
                    'text': prediction,
                    'score': round(float(score), 2)
                })
        return results

    def get_text(self, img, short_size):
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8), short_size=short_size)
        result = self.crnn_rect_with_box(np.array(img), boxes_list, score_list)
        return result


image_ocr = ImageOcr()

