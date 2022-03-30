# Vision UI

![GitHub](https://img.shields.io/badge/Python-3.8-blue)
![GitHub](https://img.shields.io/github/license/Meituan-Dianping/vision-diff)
![GitHub](https://img.shields.io/docker/cloud/build/brighthai/vision-ui)

# 简介

Vision UI 源于美团视觉测试工具，提供基于图像的UI处理和分析能力

本项目无需训练模型，[Vision-ml](https://github.com/Meituan-Dianping/vision) 项目提供RCNN训练框架

<details>
<summary>环境配置</summary>

```shell
git clone git@github.com:Meituan-Dianping/vision-ui.git --depth=1
cd vision-ui
pip3 install -r requirements.txt
```

</details>





## 特性

* 超越像素对比-[视觉对比](resources/vision_diff_cn.md)

* 基于模板匹配-[图像融合](resources/vision_merge.md)

* 预训练模型-[UI目标检测](resources/vision_infer.md)

* 视觉语义-[目标追踪](resources/vision_trace.md)

* 集成模型-[文本识别](resources/vision_text.md)


## 效果展示


### UI目标检测
| App1                    | App2                    | App3                    |
|-------------------------|-------------------------|-------------------------|
| ![](image/infer_01.png) | ![](image/infer_02.png) | ![](image/infer_03.png) |

### 视觉对比

| base                           | comparison                       | diff                                  |
| ------------------------------ | -------------------------------- | ------------------------------------- |
| ![](image/base_1.png)          | ![](image/comp_1.png)            | ![](image/diff_1.png)                 |

### 其他
[效果展示](resources/vision_show.md)


## License

This project is licensed under the [MIT](./LICENSE) 


