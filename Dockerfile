FROM python:3.6.5
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list\
    && rm -Rf /var/lib/apt/lists/*\
    && apt-get update
RUN apt-get install tesseract-ocr -y && apt-get install tesseract-ocr-chi-sim -y
COPY ./utils /vision/utils/
COPY ./requirements.txt /vision/requirements.txt
COPY ./server.py ./vision/server.py
WORKDIR /vision
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/
RUN mkdir capture\
    && pip install --upgrade pip -i ${PIP_MIRROR}\
    && pip install -r requirements.txt -i ${PIP_MIRROR}
CMD ["python", "server.py"]
