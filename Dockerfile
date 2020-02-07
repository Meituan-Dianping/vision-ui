FROM python:3.6.5
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list\
    && rm -Rf /var/lib/apt/lists/*\
    && apt-get update
RUN apt-get install tesseract-ocr -y && apt-get install tesseract-ocr-chi-sim -y
COPY ./utils /vision/utils/
COPY ./requirements.txt /vision/requirements.txt
COPY ./server.py ./vision/server.py
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/
WORKDIR ~/.cnocr
RUN pip install gdown -i ${PIP_MIRROR}\
    && gdown https://drive.google.com/uc?id=1XPS9cK468jnwhy04QKf6JcsfLa8A3yT-
WORKDIR /vision
RUN mkdir capture\
    && pip install --upgrade pip -i ${PIP_MIRROR}\
    && pip install -r requirements.txt -i ${PIP_MIRROR}
CMD ["python", "server.py"]
