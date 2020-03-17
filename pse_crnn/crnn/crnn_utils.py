import torch
from PIL import Image
import numpy as np
from torchvision import transforms


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
            _image = np.zeros((img_h, img_w), dtype='uint8')
            _image[:] = 255
            _image[:, :w] = np.array(img)
            img = Image.fromarray(_image)
        else:
            img = img.resize((img_w, img_h), self.interpolation)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        return img


class StrLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text, depth=0):
        """Support batch or single str."""
        length = []
        result = []
        for s in text:
            length.append(len(s))
            for char in s:
                index = self.dict[char]
                result.append(index)
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
