
## 服务部署

支持源代码和docker容器两种方式启动服务

### 一、源代码方式

#### 1.环境要求

python 3.6.5

#### 2.安装依赖库

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

#### 3.启动服务
```bash
python3 server.py
```


### 二、容器方式

#### 1.环境要求

[安装Docker](https://yeasy.gitbooks.io/docker_practice/install/mac.html) 

#### 2.启动容器
构建镜像

```bash
docker build -t vision-ui .

```

如果本地需要处理的图像文件在/User/image，根据实际替换为实际路径，本地使用的服务端口为9092，执行如下命令启动容器

```bash
docker run -it -d --name container_vision -p 9092:9092 -v /User/image:/vision/capture vision-ui
```
