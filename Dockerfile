FROM python:3.6.5
COPY ./utils /vision/utils/
COPY ./requirements.txt /vision/requirements.txt
COPY ./server.py ./vision/server.py
WORKDIR /vision
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/
RUN mkdir capture\
    && pip install --upgrade pip -i ${PIP_MIRROR}\
    && pip install -r requirements.txt -i ${PIP_MIRROR}
CMD ["python", "server.py"]
