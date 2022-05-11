# UI目标追踪

> Vision-trace

### 简介
鲁棒性目标查找

### 本地调试

1.CLIP环境配置
```shell
pip install git+https://github.com/openai/CLIP.git
```

2.查看区域来源，代码示例用ui-infer[目标检测](vision_infer.md)

- ui-infer预训练模型，直接识别，业务页面，位置准确速度快
- patches滑动窗口，需调整patch的w/h，适用稀疏元素，游戏页面，速度一般
- 其他方法，如提取边缘找包围框，只需提供位置信息

3.调试代码vision-ui/services/image_trace.py

- 下载[clip_vit32_feat.onnx](https://github.com/Meituan-Dianping/vision-ui/releases/download/v0.2.1/clip_vit32_feat.onnx) 到config.py文件中定义的目录
- search_target_image方法参数定义，根据实际场景，图像文本可选其一
```shell
# 图像目标系数
image_alpha = 1.0
# 文本描述系数
text_alpha = 0.6
# 最大匹配目标数量
top_k = 3
```
- 目标描述定义
```shell
# 构造目标图像
target_img = np.zeros([100, 100, 3], dtype=np.uint8)+255
cv2.putText(target_img, 'Q', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), thickness=3)
# 可直接用预制图
# target_img = cv2.imread("./capture/local_images/search_icon.png")
# 目标语言描述
desc = "shape of magnifier with blue background"
target_image_info = {'img': target_img, 'desc': desc}
```



