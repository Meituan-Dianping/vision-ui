FROM python:3.6.5
RUN git clone https://github.com/Meituan-Dianping/vision-diff.git\
    && mv vision-diff vision
WORKDIR /vision
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/
RUN mkdir capture\
    && pip install --upgrade pip -i ${PIP_MIRROR}\
    && pip install -r requirements.txt -i ${PIP_MIRROR}
CMD ["python", "server.py"]
