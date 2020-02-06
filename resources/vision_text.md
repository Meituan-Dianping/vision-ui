# Vision text

> 基于图像的文本解析

## 系统设计

算法分为两部分，定位和识别，文本识别引擎基于[cnocr](https://github.com/breezedeus/cnocr/blob/master/README_cn.md)

![](../image/image_text.png)


## 使用说明

### 环境要求

[安装Docker](https://yeasy.gitbooks.io/docker_practice/install/mac.html)

### 部署容器

如果本地需要处理的图像文件在/User/image，根据实际替换为实际路径，本地使用的服务端口为9092，执行如下命令启动容器

```bash
docker run -it -d --name container_vision -p 9092:9092 -v /User/image:/vision/capture brighthai/vision
```

通过Http协议请求服务，参数"image"表示图像文件的路径
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
