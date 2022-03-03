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
1.下载预训练的UI目标检测模型[ui-det](https://github.com/Meituan-Dianping/vision-ui/releases/download/v0.2/ui_det_v1.onnx) 到指定的目录，
修改vision-ui/services/image_infer.py文件中调试代码部分，替换model_path。

2.运行调试代码，结果文件保存在指定的infer_result_path目录