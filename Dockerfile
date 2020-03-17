FROM python:3.6.5
COPY ./api /vision/api/
COPY ./service /vision/service/
COPY ./pse_crnn /vision/pse_crnn
COPY ./requirements.txt /vision/requirements.txt
COPY ./server.py ./vision/server.py
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/
WORKDIR /root/.cnocr
RUN pip install gdown -i ${PIP_MIRROR}\
    && gdown https://drive.google.com/uc?id=1XPS9cK468jnwhy04QKf6JcsfLa8A3yT-
WORKDIR /vision
RUN mkdir capture\
    && pip install --upgrade pip -i ${PIP_MIRROR}\
    && pip install -r requirements.txt -i ${PIP_MIRROR}
CMD ["python", "server.py"]
