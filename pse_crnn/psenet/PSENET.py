import torch
from torchvision import transforms
import cv2
import numpy as np

from .pse import decode as pse_decode


class PSENetHandle(object):
    def __init__(self, model_path, net, scale, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.scale = scale
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
        self.net = torch.load(model_path, map_location=self.device)['state_dict']
        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            net.scale = scale
            try:
                sk = {}
                for k in self.net:
                    sk[k[7:]] = self.net[k]
                net.load_state_dict(sk)
            except Exception as e:
                net.load_state_dict(self.net)
            self.net = net
        self.net.eval()

    def predict(self, img: str, long_size: int = 640):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h > w:
            scale_h = long_size / h
            tar_w = w * scale_h
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            scale_w = tar_w / w
        else:
            scale_w = long_size / w
            tar_h = h * scale_w
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            scale_h = tar_h / h
        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)
        img = img.astype(np.float32)
        img /= 255.0
        img -= np.array((0.485, 0.456, 0.406))
        img /= np.array((0.229, 0.224, 0.225))
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            preds = self.net(tensor)
            preds, boxes_list, rects = pse_decode(preds[0], self.scale)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            rects_re = [] #degree, w, h, cx, cy
            if len(boxes_list):
                boxes_list = boxes_list / scale
                for rect in rects:
                    temp_rec = []
                    temp_rec.append(rect[-1])
                    temp_rec.append(rect[1][1] / scale[0] )
                    temp_rec.append(rect[1][0] / scale[1] )
                    temp_rec.append(rect[0][0] / scale[0] )
                    temp_rec.append(rect[0][1] / scale[1] )
                    rects_re.append(temp_rec)
        return preds, boxes_list, rects_re
