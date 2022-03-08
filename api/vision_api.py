import hashlib
import os
from flask import jsonify
from flask import request
from flask import Blueprint
from flask import make_response
import numpy
import requests
from config import IMAGE_INFER_MODEL_PATH
from service.image_diff import ImageDiff
from service.image_infer import ImageInfer
from service.image_merge import Stitcher
from service.image_similar import HashSimilar
from service.image_text import get_image_text
from service.image_utils import get_pop_v

vision = Blueprint('vision', __name__, url_prefix='/vision')
IMAGE_TEMP_DIR = './capture/temp'
image_infer = ImageInfer(IMAGE_INFER_MODEL_PATH)


@vision.route('/diff', methods=["POST"])
def vision_diff():
    data = {
        "code": 0,
        "data": ImageDiff().get_image_score(request.json['image1'], request.json['image2'],
                                            request.json['image_diff_name'])
    }
    return jsonify(data)


@vision.route('/merge', methods=["POST"])
def vision_merge():
    data = {
        "code": 0,
        "data": Stitcher(request.json['image_list']).image_merge(
            request.json['name'],
            without_padding=request.json.get('without_padding')
        )
    }
    return jsonify(data)


@vision.route('/similar', methods=["POST"])
def vision_similar():
    data = {
        "code": 0,
        "data": HashSimilar().get_hash_similar(request.json['image1'], request.json['image2'])
    }
    return jsonify(data)


@vision.route('/pop', methods=["POST"])
def vision_pop():
    data = {
        "code": 0,
        "data": get_pop_v(request.json['image'])
    }
    return jsonify(data)


@vision.route('/text', methods=["POST"])
def vision_text():
    data = {
        "code": 0,
        "data": get_image_text(request.json['image'])
    }
    resp = make_response(jsonify(data))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp


@vision.route('/ui-infer', methods=["POST"])
def vision_infer():
    code = 0
    data = None
    img_url = request.json['url']
    # download image
    try:
        image_name = f'{hashlib.md5(img_url.encode(encoding="utf-8")).hexdigest()}.{img_url.split(".")[-1]}'
        image_path = os.path.join(IMAGE_TEMP_DIR, image_name)
        if not os.path.exists(IMAGE_TEMP_DIR):
            os.makedirs(IMAGE_TEMP_DIR)
        r = requests.get(img_url, stream=True)
        with open(image_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32):
                f.write(chunk)
    except Exception as e:
        code = 4
        data = f'download image error, e: {e}'
    # image ui infer
    if code == 0:
        try:
            dets = image_infer.ui_infer(image_path)
            boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            # 循环，组装数据，并返回
            data = []
            for i in range(len(boxes)):
                box = boxes[i]
                box = box.tolist() if isinstance(box, (numpy.ndarray,)) else box
                type = image_infer.UI_CLASSES[int(cls_inds[i])]
                score = scores[i]
                data.append(
                    {
                        "elem_det_type": "image" if type == 'pic' else type,
                        "elem_det_region": box,
                        "probability": score
                    }
                )
        except Exception as e:
            code = 5
            data = f'ui infer error, e: {e}'
        finally:
            os.remove(image_path)
    result = {
        "code": code,
        "data": data
    }
    resp = make_response(jsonify(result))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp
