import cv2
import torch
import clip
import numpy as np
from PIL import Image
from scipy import spatial
from service.image_infer import get_ui_infer
from service.image_utils import get_roi_image, img_show


def cosine_similar(l1, l2):
    return 1 - spatial.distance.cosine(l1, l2)


class ImageTrace(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}.")
        print("Downloading model will take a while for the first time.")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def search_image(self, target_image_info: dict, source_image_path: str):
        top_k = 3  # 最大匹配数量
        text_alpha = 0.5  # 文本语义scale
        roi_list = []
        img_text_score = []
        target_image_path = target_image_info.get('path', '')
        target_image_desc = target_image_info.get('desc', '')
        target_image = cv2.imread(target_image_path)
        source_image = cv2.imread(source_image_path)
        image_infer_result = get_ui_infer(source_image_path)
        text = clip.tokenize([target_image_desc]).to(self.device)
        # 提取检测目标
        for roi in image_infer_result:
            x1, y1, x2, y2 = list(map(int, roi['elem_det_region']))
            roi = get_roi_image(source_image, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_list.append(self.preprocess(img_pil).to(self.device))
        # 计算图像和文本匹配向量
        with torch.no_grad():
            img_pil = Image.fromarray(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
            target_image_input = torch.tensor(self.preprocess(img_pil).unsqueeze(0).to(self.device))
            target_image_features = self.model.encode_image(target_image_input)
            source_image_input = torch.tensor(np.stack(roi_list))
            source_image_features = self.model.encode_image(source_image_input)
            logits_per_image, logits_per_text = self.model(source_image_input, text)
            probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        # 图像加文本
        for i, source_image_feature in enumerate(source_image_features):
            score = cosine_similar(target_image_features[0], source_image_feature)
            img_text_score.append(score + probs[0][i]*text_alpha)
        score_norm = (img_text_score - np.min(img_text_score)) / (np.max(img_text_score) - np.min(img_text_score))
        top_k_ids = np.argsort(score_norm)[-top_k:]
        return top_k_ids, score_norm, image_infer_result


if __name__ == '__main__':
    target_image_info = {
        'path': "./capture/local_images/search_icon.png",
        'desc': "shape of magnifier with blue background"
    }
    source_image_path = "./capture/image_1.png"
    image_trace = ImageTrace()
    top_k_ids, scores, infer_result = image_trace.search_image(target_image_info, source_image_path)
    # show result
    trace_result_path = "./capture/local_images/trace_result.png"
    cls_ids = np.zeros(len(top_k_ids), dtype=int)
    boxes = [infer_result[i]['elem_det_region'] for i in top_k_ids]
    scores = [float(scores[i]) for i in top_k_ids]
    image_trace_show = img_show(cv2.imread(source_image_path), boxes, scores, cls_ids, conf=0.5, class_names=['T'])
    cv2.imwrite(trace_result_path, image_trace_show)
