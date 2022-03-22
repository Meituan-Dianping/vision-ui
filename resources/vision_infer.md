# UI目标检测

> Vision-infer

### 简介
在CPU下能快速推理的UI检测模型


### 模型性能

* 基于[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 目标检测框架，训练阶段修改了部分超参数，
识别目标为UI中常见的图片和图标，文本可由OCR获得详见[文本识别](vision_text.md)，在开放测试集中平均准确超过90%


* 经[ONNX](https://onnx.ai) Optimizer转换，用i7-9750H CPU推理时间105ms，
可转为[TensorRT](https://github.com/onnx/onnx-tensorrt) 用GPU进一步加速推理

### 使用说明

1.下载预训练的UI目标检测模型[ui-det-v2](https://github.com/Meituan-Dianping/vision-ui/releases/download/v0.2/ui_det_v2.onnx) 到指定的目录，
修改vision-ui/config.py文件，替换IMAGE_INFER_MODEL_PATH。

2.运行vision-ui/services/image_infer.py调试代码，结果文件保存在指定的infer_result_path目录，查看效果。

3.[启动服务](launch_service.md)

4.通过Http协议请求，接口定义如下：

  GET  /vision/ui-infer
  参数列表：

  | 参数   | 类型     |   说明   | 是否必填  |
  | :---: | :-----: | :------ | :--------- |
  |type | String | 上传图片的方式 | 非必填，可填“url”或“base64”，默认“url”，即使用url的方式 |
  |url  | String | 当type为“url”时，表示图片的下载链接| type不填或为“url”时，必填|
  |image| String | 当type为“base64”时，表示图片数据base64编码后的字符串 | type为“base64”时，必填 |

  ##### 以图片url方式调用示例
  ```bash
  curl -H "Content-Type:application/json" -X POST --data '{"url":"http://XXX/imagename.png"}' http://localhost:9092/vision/ui-infer
  ```
  服务返回
  ```bash
  {
    "code":0, 
    "data":[
      {
          'elem_det_region': [41.2695, 118.5093, 67.1008, 162.8195], # 矩形方框表示 [x1, y1, x2, y2]
          'elem_det_type': 'icon', # 元素类型，icon/image
          'probability': 0.9590860605239868
      }
    ]
  }
  ```

  ##### 以图片base64方式调用示例

  以python为例，图片base64处理代码如下：
  ```python
  import base64
  import cv2

  pic_path = 'image.png'
  image = cv2.imread(pic_path)
  image_b = base64.b64encode(cv2.imencode('.png', image)[1].tostring())
  image_source = bytes.decode(image_b)
  ```

  ```bash
  curl -H "Content-Type:application/json" -X POST --data '{"type":"base64", "image":"image_source"}' http://localhost:9092/vision/ui-infer
  ```
  服务返回
  ```bash
  {
    "code":0, 
    "data":[
      {
          'elem_det_region': [41.2695, 118.5093, 67.1008, 162.8195], # 矩形方框表示 [x1, y1, x2, y2]
          'elem_det_type': 'icon', # 元素类型，icon/image
          'probability': 0.9590860605239868
      }
    ]
  }
  ```