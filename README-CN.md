![](image/vision_diff.png)

![GitHub](https://img.shields.io/badge/Python-3.6-blue)
![GitHub](https://img.shields.io/github/license/Meituan-Dianping/vision-diff)

> 基于行特征的图像对比算法

**Vision-diff** 是一种可实现类似 [diff utility](https://en.wikipedia.org/wiki/Diff)效果的图像对比算法. 
和基于像素对比不一样的是，算法基于行特征计算和对比，可实现更加清晰和更接近语义的图像对比展示。

![](image/image_4.png)

![](image/image_5.png)

Vision-diff 主要的目标是标记两个版本的截图差异,对移动端的使用场景专做了对应优化。


### 容器服务

Vision diff 推进通过docker部署容器的方式使用服务

[容器服务](./container-service.md)

## 使用方法

### 配置环境

Python3.5 or higher

```bash
pip install  -r requirements.txt
```

### 代码调试

- image_diff() 使用3个传入参数，前两个参数是参与对比的两个图像文件的访问路径
- 第三个参数是对比结果图像的输出路径
- 对比图的标记部分会使用红色着色

```python
from image_diff import ImageDiff

img = ImageDiff()
img.increment_diff("1.png", "2.png", "output_img.png")
```

算法的默认配置参数推荐使用移动设备的原始分辨率图像，多数为1080p

"image_diff.py" 有一些默认的高级参数可以根据你的实际使用需求调整:
  - "w" 表示滤波器的行数宽度，较高的数值会过滤更多的非连续标记行数
  - "padding" 表示处理"图像融合"的图像右边label
  - "h_scale" 表示需要屏蔽图像顶部设备状态信息的长度y0和图像宽的比值，这里h_scale=y0/w
  - "hash_score" 表示行特征计算匹配分数的阈值
  - "pixel_value" 表示标记像素的参考值


### 图像结构相似分数

Vision-diff的计算时间是O(ND)的，为了获得更好的性能和效果的平衡，你可以在进行对比之前进行图像结构相似分数计算

```python
from image_similar import HashSimilar

score = HashSimilar.get_similar("1.png", "2.png")
```

score 有3个值，一般建议值为0.8的时候进行Vision-diff计算

  - 1.0: 图像内容和分布一样
  - 0.8: 图像内容和分布部分不一样
  - 0.2: 图像不是来自同一页面


## 图像增量对比
传统的基于Pixel by Pixel的像素值对比可以标记图像在同一位置的内容不同

![](image/image_1.png)

如果图像B的内容做了一点平移，像素对比会从内容平移的位置进行标记，不能实现我们对相同内容保持一致的需求

![](image/image_2.png)

Vision diff 可以更好的处理内容纵向上的分布平移，只标记增量的内容部分 

![](image/image_3.png)

### 移动端UI测试
- 有时我们会需要两张来自不同App版本的页面截图进行对比来发现UI上的更新和可能的问题。比如UI兼容性测试，如果
使用传统的基于像素的对比算法，一部分内容的位置平移会标记对比图的基本所有部分

![](image/image_4.png)

- 使用Vision-diff，图像的对比结果更接近使用者的语义，只需要关注内容真正改变的部分

![](image/image_5.png)


## 算法性能

在Corei7@2.2Ghz的计算条件下一般需要2-6s，具体取决于图像不同的内容部分占比

## 参考

图像的转换路径算法参考了[论文](http://www.xmailserver.org/diff2.pdf).
