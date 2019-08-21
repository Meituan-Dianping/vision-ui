from flask import Flask
from flask import jsonify
from flask import request
from utils.image_diff import ImageDiff
from utils.image_merge import Stitcher
from utils.image_similar import HashSimilar
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/vision/diff', methods=["POST"])
def vision_diff():
    data = {
        "code": 0,
        "data": ImageDiff().get_image_score(request.json['image1'], request.json['image2'],
                                            request.json['image_diff_name'])
    }
    return jsonify(data)


@app.route('/vision/merge', methods=["POST"])
def vision_merge():
    data = {
        "code": 0,
        "data": Stitcher(request.json['image_list']).image_merge(request.json['name'])
    }
    return jsonify(data)


@app.route('/vision/similar', methods=["POST"])
def vision_similar():
    data = {
        "code": 0,
        "data": HashSimilar().get_hash_similar(request.json['image1'], request.json['image2'])
    }
    return jsonify(data)


@app.errorhandler(Exception)
def error(e):
    ret = dict()
    ret["code"] = 1
    ret["data"] = repr(e)
    return jsonify(ret)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9092)
