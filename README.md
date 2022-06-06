# Vision UI

![GitHub](https://img.shields.io/badge/Python-3.8-blue)
![GitHub](https://img.shields.io/github/license/Meituan-Dianping/vision-diff)
![GitHub](https://img.shields.io/docker/cloud/build/brighthai/vision-ui)

## 简介

Vision UI 源于美团视觉测试工具，提供基于图像的UI分析能力

本项目无需训练模型，[Vision-ml](https://github.com/Meituan-Dianping/vision) 项目提供RCNN训练框架

环境配置

```shell
# create venv and activate
git clone git@github.com:Meituan-Dianping/vision-ui.git --depth=1
cd vision-ui
pip3 install -r requirements.txt
# set working directory for command-line usage
export PYTHONPATH=$PYTHONPATH:$path/to/project/vision-ui
```


## 特性

* 超越像素对比-[视觉对比](resources/vision_diff_cn.md)

* 基于模板匹配-[图像融合](resources/vision_merge.md)

* 预训练模型-[UI目标检测](resources/vision_infer.md)

* 视觉语义-[语义目标识别](resources/vision_trace.md)

* 集成模型-[文本识别](resources/vision_text.md)


## 效果展示


### UI目标检测
| App1                    | App2                    | App3                    |
|-------------------------|-------------------------|-------------------------|
|![infer_01](https://user-images.githubusercontent.com/11002329/169336590-c0a8d6b9-a4cf-4449-8c84-9444c89f63de.png)|![infer_02](https://user-images.githubusercontent.com/11002329/169336682-2640827f-aba3-4f62-8baf-ccfb4a0f3e2a.png)|![infer_03](https://user-images.githubusercontent.com/11002329/169336771-347cdd14-e332-404f-b361-200f303c50fb.png)|


### 语义目标识别
| image or text query  | proposal backend | result                               |
|----------------------|-----------------|--------------------------------------|
| <img width="45" alt="mario" src="https://user-images.githubusercontent.com/11002329/169337384-ba2763c1-3a5f-4161-adce-27d6b58e2a80.png">| ui-infer    |![trace_result_mario](https://user-images.githubusercontent.com/11002329/169337586-0c1405ea-6dc1-4f27-a6a2-4c294730f1c7.png) |
| a toy dinosaur       | patches         |![trace_result_dinosaur](https://user-images.githubusercontent.com/11002329/169338047-702214ab-b0fb-43ff-bdd3-b6746539a14e.png)|
|  mario | patches     | ![mario_jump](https://user-images.githubusercontent.com/11002329/172109580-df200eda-ac05-484e-8ce0-6607f3c2f5f8.gif)|

### 更多
[效果展示](resources/vision_show.md)


## License

This project is licensed under the [MIT](./LICENSE) 


