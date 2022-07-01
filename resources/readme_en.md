# Vision UI

![GitHub](https://img.shields.io/badge/Python-3.8-blue)
![GitHub](https://img.shields.io/github/license/Meituan-Dianping/vision-diff)
![GitHub](https://img.shields.io/docker/cloud/build/brighthai/vision-ui)

## Introduction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sXcgUav3xo9yCFEbqIK4-QOvnMZ3A1hj?usp=sharing)

Vision UI was provided by Meituan-Vision-Testing tool.

There is no need to train model for this project，[Vision-ml](https://github.com/Meituan-Dianping/vision) is an RNN training project


Setup

```shell
# create venv and activate
git clone git@github.com:Meituan-Dianping/vision-ui.git --depth=1
cd vision-ui
pip3 install -r requirements.txt
# set working directory for command-line usage
export PYTHONPATH=$PYTHONPATH:$path/to/project/vision-ui
```


## Features

* beyond pixel diff-[vision diff](vision_diff_cn.md)

* template match-[image merge](vision_merge.md)

* pre-trained model-[UI detection](vision_infer.md)

* vision semantic-[semantic target recognition](vision_trace.md)

* pre-trained model-[OCR](vision_text.md)


## Preview


### UI detection
| App1                    | App2                    | App3                    |
|-------------------------|-------------------------|-------------------------|
|![infer_01](https://user-images.githubusercontent.com/11002329/169336590-c0a8d6b9-a4cf-4449-8c84-9444c89f63de.png)|![infer_02](https://user-images.githubusercontent.com/11002329/169336682-2640827f-aba3-4f62-8baf-ccfb4a0f3e2a.png)|![infer_03](https://user-images.githubusercontent.com/11002329/169336771-347cdd14-e332-404f-b361-200f303c50fb.png)|


### semantic target recognition
| image or text query  | proposal backend | result                               |
|----------------------|-----------------|--------------------------------------|
| <img width="45" alt="mario" src="https://user-images.githubusercontent.com/11002329/169337384-ba2763c1-3a5f-4161-adce-27d6b58e2a80.png">| ui-infer    |![trace_result_mario](https://user-images.githubusercontent.com/11002329/169337586-0c1405ea-6dc1-4f27-a6a2-4c294730f1c7.png) |
| a toy dinosaur       | patches         |![trace_result_dinosaur](https://user-images.githubusercontent.com/11002329/169338047-702214ab-b0fb-43ff-bdd3-b6746539a14e.png)|
|  mario | patches     | ![mario_jump](https://user-images.githubusercontent.com/11002329/172109580-df200eda-ac05-484e-8ce0-6607f3c2f5f8.gif)|

### More
[preview](vision_show.md)


## License

This project is licensed under the [MIT](./LICENSE) 


