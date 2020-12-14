## 图像融合

> Vision-merge

只需在设备上获得连续的几张截图，就能获得还原页面的实际空间展示。在页面上截图后向上滑动30%屏幕长度的距离再截图，重复上述步骤获得几张前后连续的图像

### 启动服务

[启动服务](launch_service.md)


### 使用说明
通过HTTP请求服务，参数image_list表示图片在capture文件夹下的相对路径，name表示融合后图像的存放路径

```bash
curl -H "Content-Type:application/json" -X POST --data '{
  "image_list":["0_1.png","0_2.png","0_3.png"],
  "name":"image_merge.png",
}' http://localhost:9092/vision/merge
```
服务返回
```bash
{
  "code":0, #值范围[0-正常,1-服务错误]
  "data":"image_merge.png", #融合后图像的保存路径,和传入的“name”参数一样
}
```