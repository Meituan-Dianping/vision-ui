# Vision text

> 基于图像的文本组件解析


<img width="600" src="../image/vision_text_1.png"/>

<img width="600" src="../image/vision_text_2.png"/>

文本识别模型基于psenet+crnn的[chinese-ocr-lite](https://github.com/ouyanghuiyu/chineseocr_lite)，对移动端的场景做了部分参数和模型的调整


## 使用说明

### 启动服务

 [启动服务](container_service.md)

### 请求参数

> 通过Http协议请求，参数"image"表示图像文件在"capture"下的路径
```bash
curl -H "Content-Type:application/json" -X POST --data '{
  "image":"image_1.png"
}' http://localhost:9092/vision/text
```
服务返回
```bash
{
  "code":0, 
  "data":[
    {
      "pos": [100,200], #表示可点击的坐标
      "text": "用户使用说明"  #图像解析到的文本内容
    },{
      "pos": [300,500],
      "text": "同意"
    }
  ]
}
```
