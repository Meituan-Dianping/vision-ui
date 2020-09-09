from PIL import Image
import numpy as np


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        size = self.size
        img_w, img_h = size
        scale = img.size[1] * 1.0 / img_h
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, img_h), self.interpolation)
        w, h = img.size
        if w <= img_w:
            tmp_img = np.zeros((img_h, img_w), dtype='uint8')
            tmp_img[:] = 255
            tmp_img[:, :w] = np.array(img)
            img = Image.fromarray(tmp_img)
        else:
            img = img.resize((img_w, img_h), self.interpolation)
        img = np.array(img,dtype=np.float32)
        img -= 127.5
        img /= 127.5
        img = img.reshape([*img.shape,1])
        return img


class StringLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)
