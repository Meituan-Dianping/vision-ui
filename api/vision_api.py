import hashlib
import os
from flask import jsonify
from flask import request
from flask import Blueprint
from flask import make_response
from service.image_diff import ImageDiff
from service.image_infer import get_ui_infer
from service.image_merge import Stitcher
from service.image_similar import HashSimilar
from service.image_text import get_image_text
from service.image_utils import download_image, get_pop_v, save_base64_image


vision = Blueprint('vision', __name__, url_prefix='/vision')


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
    image_type = request.json.get('type', 'url')
    if image_type == 'url':
        img_url = request.json['url']
        image_name = f'{hashlib.md5(img_url.encode(encoding="utf-8")).hexdigest()}.{img_url.split(".")[-1]}'
        success, image_path, message = download_image(img_url, image_name)
    elif image_type == 'base64':
        base64_image = request.json['image']
        image_name = f'{hashlib.md5(base64_image.encode(encoding="utf-8")).hexdigest()}.png'
        success, image_path, message = save_base64_image(base64_image, image_name)
    else:
        success = False
        message = f'ui infer not support this type: {image_type}'
    
    if success:
        try:
            data = get_ui_infer(image_path)
        except Exception as e:
            code = 5
            data = f'ui infer error, e: {e}'
        finally:
            os.remove(image_path)
    else:
        code = 4
        data = message
    result = {
        "code": code,
        "data": data
    }
    resp = make_response(jsonify(result))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp
