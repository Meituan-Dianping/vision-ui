FROM python:3.6.5
COPY ./api /vision/api/
COPY ./service /vision/service/
COPY ./dbnet_crnn /vision/dbnet_crnn
COPY ./requirements.txt /vision/requirements.txt
COPY ./server.py ./vision/server.py
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/
WORKDIR /vision
RUN mkdir capture\
    && pip install --upgrade pip -i ${PIP_MIRROR}\
    && pip install -r requirements.txt -i ${PIP_MIRROR}
CMD ["python3", "server.py"]
