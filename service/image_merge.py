import numpy
import cv2


class Stitcher(object):
    def __init__(self, pictures):
        self.img_list = pictures

    @staticmethod
    def add_padding(img, w):
        h, _, _ = img.shape
        padding = numpy.zeros((h, w, 3), numpy.uint8) + 210
        return numpy.hstack((img, padding))

    @staticmethod
    def merge_with_param(img1, img2, w, roi_scale, tail_scale, index):
        img1 = img1.copy()
        img2 = img2.copy()
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        img1 = img1[0:h1 - int(tail_scale * h2), :, ]
        if w1 == w2:
            img1 = Stitcher.add_padding(img1, w)
        h1, w1, _ = img1.shape
        _img1 = img1[:, :w1 - w, :]
        h1, w1, _ = _img1.shape
        if w1 != w2:
            raise Exception("Image merge: different width")
        roi_y = int(h2 * roi_scale)
        roi = _img1[(h1 - roi_y):h1, :, :]
        res = cv2.matchTemplate(img2, roi, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        std = numpy.std(roi)
        if std < 10:
            max_val = 0.95
        cut_point = (max_loc[0], max_loc[1] + int(roi_scale * h2))
        img2 = Stitcher.add_padding(img2, w)
        img = numpy.vstack((img1, img2[cut_point[1]:h2, :, :]))
        cv2.putText(img, "-" + str(index), (w2, h1), cv2.FONT_ITALIC, 1.5, (123, 189, 60), 5)
        return img, max_val

    @staticmethod
    def stack_image(img1, img2, w, index):
        img1 = img1.copy()
        img2 = img2.copy()
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        if w1 == w2:
            img1 = Stitcher.add_padding(img1, w)
        h1, w1, _ = img1.shape
        _img1 = img1[:, :w1 - w, :]
        h1, w1, _ = _img1.shape
        if w1 != w2:
            raise Exception("Image width different")
        img2 = Stitcher.add_padding(img2, w)
        img = numpy.vstack((img1, img2))
        cv2.putText(img, "-" + str(index), (w2, h1), cv2.FONT_ITALIC, 1.5, (123, 189, 60), 5)
        return img

    @staticmethod
    def img_merge(img1, img2, index, merge=True):
        w = 80
        match = 0.98
        img_list = []
        score_list = []
        scale_list = [
            {
                "roi_scale": 0.12,
                "tail_scale": 0.18
            }, {
                "roi_scale": 0.08,
                "tail_scale": 0.32
            }, {
                "roi_scale": 0.08,
                "tail_scale": 0.08
            }, {
                "roi_scale": 0.05,
                "tail_scale": 0.2
            }, {
                "roi_scale": 0.1,
                "tail_scale": 0.4
            }, {
                "roi_scale": 0.08,
                "tail_scale": 0.15
            }
        ]
        if merge:
            for scale in scale_list:
                img, score = Stitcher.merge_with_param(img1, img2, w, scale["roi_scale"], scale["tail_scale"], index)
                img_list.append(img)
                score_list.append(score)
                if score > match:
                    break
                if scale == scale_list[-1] and max(score_list) < 0.92:
                    img = Stitcher.stack_image(img1, img2, w, index)
                else:
                    img = img_list[score_list.index(max(score_list))]
        else:
            img = Stitcher.stack_image(img1, img2, w, index)
        return img

    def image_merge(self, name, merge=True):
        """
        :param name: image path to save after merge
        :param merge: operate image merge
        :return: image merge name
        """
        img_list = []
        for img in self.img_list:
            img_list.append('capture/'+img)
        name = 'capture/'+name
        if len(img_list) < 2:
            cv2.imwrite(name, cv2.imread(img_list[0]))
        else:
            img1 = img_list[0]
            for img in img_list[1:]:
                index = img_list.index(img)
                if img_list.index(img) == 1:
                    img1 = self.img_merge(cv2.imread(img1), cv2.imread(img), index, merge)
                else:
                    img1 = self.img_merge(img1, cv2.imread(img), index, merge)
            img_merge = img1
            cv2.imwrite(name, img_merge)
        return name
