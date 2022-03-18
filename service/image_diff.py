import cv2
from service.image_similar import HashSimilar
from service.image_utils import get_hash_score, m_diff


class LineFeatureEqual(object):
    def __init__(self):
        self.thresh = 0.85

    def equal(self, a, b):
        if get_hash_score(a, b) > self.thresh:
            return True
        else:
            return False


class ImageDiff(object):
    def __init__(self, w=9, padding=80, w_scale=850, h_scale=0.08, pixel_value=28):
        self.filter_w = w
        self.padding = padding
        self.size_scale = w_scale
        self.head_scale = h_scale
        self.pixel_value = pixel_value

    @staticmethod
    def get_line_list(op_list):
        """
        get line list
        :param op_list: op list
        :return: line list
        """
        line1_list = []
        line2_list = []
        for op in op_list:
            if op["operation"] == "insert":
                line1_list.append(op["position_new"])
            if op["operation"] == "delete":
                line2_list.append(op["position_old"])
        return line1_list, line2_list

    @staticmethod
    def get_line_feature(image, precision=8):
        """
        get line feature of input image
        :param image: image in numpy shape
        :param precision: feature precision
        :return: line feature
        """
        line_feature = []
        for y in range(image.shape[0]):
            img = cv2.resize(image[y], (precision, precision))
            img_list = img.flatten()
            avg = sum(img_list) * 1. / len(img_list)
            avg_list = ["0" if i < avg else "1" for i in img_list]
            line_feature.append([int(''.join(avg_list[x:x+4]), 2) for x in range(0, precision*precision)])
        return line_feature

    def get_image_feature(self, img1, img2):
        """
        get image feature with padding processing
        :param img1: imageA in numpy shape
        :param img2: imageB in numpy shape
        :return: image feature
        """
        h1, w = img1.shape
        img1 = img1[:, :w-self.padding]
        img2 = img2[:, :w-self.padding]
        img1_feature = self.get_line_feature(img1)
        img2_feature = self.get_line_feature(img2)
        return img1_feature, img2_feature

    def line_filter(self, line_list):
        """
        calculate line list with param
        :param line_list: line list
        :return: filtered line list
        """
        i = 0
        w = self.filter_w
        line = []
        while i < len(line_list)-w-1:
            f = line_list[i:i+w]
            s = 0
            for j in range(w-1):
                s = s + f[j+1] - f[j]
            if s - w <= 6:
                for l in f:
                    if l not in line:
                        line.append(l)
            i = i + 1
        return line

    def get_image(self, image_file):
        """
        cv2.read image and 3d to 1d
        :param image_file: image file path
        :return: image in numpy shape
        """
        image = cv2.imread(image_file)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        h, w = img.shape
        scale = self.size_scale/w
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return img

    @staticmethod
    def get_pixel(img, x, y):
        """
        get pixel value of image
        :param img: image in numpy shape
        :param x: position x
        :param y: position y
        :return: pixel value
        """
        h, w = img.shape
        p = 0
        if y < h:
            p = img[y][x]
        return p

    def increment_diff(self, image1, image2, image_show) -> int:
        """
        calculate increment image diff
        :param image1: input image A
        :param image2: input image B
        :param image_show: increment diff image for show
        :return: points length of image show
        """
        img1 = self.get_image(image1)
        img2 = self.get_image(image2)
        score_list = HashSimilar.get_attention(img1, img2)
        img1_feature, img2_feature = self.get_image_feature(img1, img2)
        line1, line2 = self.get_line_list(m_diff(img1_feature, img2_feature, equal_obj=LineFeatureEqual()))
        line = line1 + line2
        line = self.line_filter(line)
        img_show = img2.copy() if img2.shape[0] > img1.shape[0] else img1.copy()
        (h, w) = img_show.shape
        img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
        points = []
        line_attention = []
        for l in line:
            i = int((len(score_list) * (l - 1) / h))
            i = 0 if i < 0 else i
            if score_list[i] < 0.98:
                line_attention.append(l)
        line = line_attention
        for y in range(int(h*0.95)):
            if y > int(w * self.head_scale):
                if y in line:
                    for x in range(w-self.padding):
                        p1 = int(self.get_pixel(img1, x, y))
                        p2 = int(self.get_pixel(img2, x, y))
                        if abs(p1 - p2) < self.pixel_value:
                            pass
                        else:
                            points.append([x, y])
        for point in points:
            cv2.circle(img_show, (point[0], point[1]), 1, (0, 0, 255), -1)
        cv2.imwrite(image_show, img_show)
        return len(points)

    def get_image_score(self, image1, image2, image_diff_name):
        score = HashSimilar.get_attention_similar('capture/'+image1, 'capture/'+image2)
        if score < 1.0:
            if score > 0.2:
                points_size = self.increment_diff('capture/'+image1, 'capture/'+image2, 'capture/'+image_diff_name)
                if points_size < 50:
                    score = 1.0
        return score
