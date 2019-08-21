import cv2
import numpy


class HashSimilar(object):
    @staticmethod
    def perception_hash(img_gray, precision=64):
        """
        get hash code of image
        """
        img_scale = cv2.resize(img_gray, (precision, precision))
        img_list = img_scale.flatten()
        avg = sum(img_list)*1./len(img_list)
        avg_list = ['0' if i < avg else '1' for i in img_list]
        return [int(''.join(avg_list[x:x+4]), 2) for x in range(0, precision*precision)]

    @staticmethod
    def get_image(img_file):
        """
        get image in numpy shape
        """
        img = cv2.imread(img_file)
        h, w, _ = img.shape
        img = img[int(w * 0.078):, :, :]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    @staticmethod
    def get_image_list(img_gray):
        """
        get image in list shape
        """
        i = 0
        img_list = []
        h, w = img_gray.shape
        stride = int(w*0.05)
        while i < h:
            img_list.append(img_gray[i:i + stride, :])
            i = i + stride
        return img_list

    @staticmethod
    def hamming_dist(s1, s2):
        assert len(s1) == len(s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

    @staticmethod
    def get_attention(img1, img2):
        """
        calculate images attention score list
        :param img1: input image A
        :param img2: input image B
        :return: score list
        """
        img1_list = HashSimilar.get_image_list(img1)
        img2_list = HashSimilar.get_image_list(img2)
        l = min(len(img1_list), len(img2_list))
        score_list = []
        for i in range(0, l, 1):
            start = i - 5 if i - 5 >= 0 else 0
            img_stack = numpy.vstack(img2_list[start:i+5])
            res = cv2.matchTemplate(img1_list[i], img_stack, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            score_list.append(max_val)
        return score_list

    @staticmethod
    def get_attention_similar(image1, image2) -> float:
        """
        calculate image content similar
        :param image1: input image A
        :param image2: input image B
        :return: similar score
            - 0.2 : image come from different schema
            - 0.8 : image has increment content
            - 1.0 : imageA is similar to imageB
        """
        img1 = HashSimilar.get_image(image1)
        img2 = HashSimilar.get_image(image2)
        std1 = numpy.std(img1)
        std2 = numpy.std(img2)
        img1_list = HashSimilar.get_image_list(img1)
        img2_list = HashSimilar.get_image_list(img2)
        l = min(len(img1_list), len(img2_list))
        score_list = HashSimilar.get_attention(img1, img2)
        score_list.sort()
        if score_list[int(len(score_list) * 0.8) - 1] < 0.8 or max(len(img1_list), len(img2_list)) - l > l \
                or abs(std1 - std2) > 35:
            return 0.2
        if min(score_list) > 0.99:
            return 1.0
        return 0.8

    @staticmethod
    def get_hash_similar(image1, image2) -> float:
        """
        :param image1: input image A
        :param image2: input image B
        :return: score of hash calculation
        """
        img1 = HashSimilar.get_image('capture/'+image1)
        img2 = HashSimilar.get_image('capture/'+image2)
        hash1 = HashSimilar.perception_hash(img1)
        hash2 = HashSimilar.perception_hash(img2)
        score = 1 - HashSimilar.hamming_dist(hash1, hash2) * 1.0 / (64 * 64)
        return score
