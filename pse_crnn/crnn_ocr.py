from pse_crnn.crnn.crnn import CRNNHandle
from pse_crnn.crnn.crnn_lite import CRNNLite
from pse_crnn.psenet.model import PSENet
from pse_crnn.psenet.PSENET import PSENetHandle
from pse_crnn.ocr_utils import crop_rect
from PIL import Image
from pse_crnn.crnn.keys import alphabetChinese as alphabet
import numpy as np


class CrnnOcr(object):
    def __init__(self, gpu_id=None, pse_scale=1):
        mobile_net = PSENet(backbone='mobilenetv2', result_num=6, scale=pse_scale)
        self._text_handle = PSENetHandle('pse_crnn/models/psenet_lite_mbv2.pth', mobile_net, pse_scale, gpu_id=gpu_id)
        crnn_net = CRNNLite(32, 1, len(alphabet) + 1, 256, n_rnn=2, leakyRelu=False, LSTM_flag=True)
        self._crnn_handle = CRNNHandle('pse_crnn/models/crnn_lite_lstm_dw_v2.pth', crnn_net, gpu_id=gpu_id)

    def _crnn_recognize(self, im, rects_re, f=1.0):
        """
        Args:
            im: image in cv2 array
            rects_re: rects
            f: scale for rect
        Returns: recognize results
        """
        results = []
        im = Image.fromarray(im)
        for index, rect in enumerate(rects_re):
            degree, w, h, cx, cy = rect
            roi = crop_rect(im,  ((cx, cy), (h, w), degree))
            roi_w, roi_h = roi.size
            roi_array = np.uint8(roi)
            if roi_h > 1.5 * roi_w:
                roi_array = np.rot90(roi_array, 1)
            roi = Image.fromarray(roi_array).convert("RGB")
            roi_ = roi.convert('L')
            try:
                predicted_text = self._crnn_handle.predict(roi_)
            except Exception as e:
                print(e)
                continue
            if predicted_text.strip() != u'':
                results.append({'pos': [int(cx * f), int(cy * f)], 'text': predicted_text})
        return results

    def crnn_ocr(self, img, pse_long_size):
        preds, boxes_list, rects_re = self._text_handle.predict(img, long_size=pse_long_size)
        result = self._crnn_recognize(np.array(img), rects_re)
        return result
