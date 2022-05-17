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
from service.image_trace import image_trace
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
    image_type = request.json.get('type', 'url')
    cls_thresh = request.json.get('cls_thresh', 0.5)
    if image_type == 'url':
        img_url = request.json['url']
        image_name = f'{hashlib.md5(img_url.encode(encoding="utf-8")).hexdigest()}.{img_url.split(".")[-1]}'
        image_path = download_image(img_url, image_name)
    elif image_type == 'base64':
        base64_image = request.json['image']
        image_name = f'{hashlib.md5(base64_image.encode(encoding="utf-8")).hexdigest()}.png'
        image_path = save_base64_image(base64_image, image_name)
    else:
        raise Exception(f'UI infer API don`t support this type: {image_type}')

    try:
        data = get_ui_infer(image_path, cls_thresh)
    finally:
        os.remove(image_path)

    result = {
        "code": code,
        "data": data
    }
    resp = make_response(jsonify(result))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp


@vision.route('/semantic-search', methods=["POST"])
def semantic_search():
    """
    req json:
    {
        type: [optional] 'url' or 'base64', 'url'
        target_image,
        source_image,
        target_desc,
        image_alpha [float] 0.0-1.0,
        text_alpha: [float] 0.0-1.0,
        top_k: [optional] integer value >=1, 1
        proposal_provider: [optional] 'ui-infer' or 'patches', 'ui-infer'
    }
    """
    code = 0
    image_type = request.json.get('type', 'url')
    target_image = request.json.get('target_image')
    source_image = request.json.get('source_image')
    if image_type == 'url':
        image_name = f'{hashlib.md5(target_image.encode(encoding="utf-8")).hexdigest()}.{target_image.split(".")[-1]}'
        target_image_path = download_image(target_image, image_name)
        image_name = f'{hashlib.md5(source_image.encode(encoding="utf-8")).hexdigest()}.{source_image.split(".")[-1]}'
        source_image_path = download_image(source_image, image_name)
    elif image_type == 'base64':
        image_name = f'{hashlib.md5(target_image.encode(encoding="utf-8")).hexdigest()}.png'
        target_image_path = save_base64_image(target_image, image_name)
        image_name = f'{hashlib.md5(source_image.encode(encoding="utf-8")).hexdigest()}.png'
        source_image_path = save_base64_image(source_image, image_name)
    else:
        raise Exception(f'Not supported type: {image_type}')
    try:
        target_image_info = {'img': target_image_path, 'desc': request.json.get('target_desc', '')}
        top_k_ids, scores, infer_result, max_confidence = image_trace.search_image(
            target_image_info, source_image_path, request.json.get('top_k', 1), request.json.get('image_alpha'),
            request.json.get('text_alpha'), request.json.get('proposal_provider', 'ui-infer')
        )
        data = []
        for i in top_k_ids:
            data.append({
                'score': round(float(scores[i]), 2),
                'boxes': [int(k) for k in infer_result[i]['elem_det_region']]
            })
    finally:
        os.remove(target_image_path)
        os.remove(source_image_path)
    result = {
        "code": code,
        "data": {
            'max_confidence': max_confidence,
            'search_result': data
        }
    }
    resp = make_response(jsonify(result))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp
