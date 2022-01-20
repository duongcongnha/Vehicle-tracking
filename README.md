# Vehicle tracking with Yolov5 + Deep Sort with PyTorch




## Before you run the tracker

Make sure that you fulfill all the requirements: Python 3.7.12 or later with all [requirements.txt](https://github.com/duongcongnha/ppattention-intermediary/blob/main/requirements.txt) dependencies installed. To install, run:

`pip install -r requirements.txt`
<br></br>
    if you have problem with `pip install dlib`, try using `conda`
<br></br>
    if you have CUDA, modified two lines `torch>=1.7.0` and `torchvision>=0.8.1` in `requirements.txt` or install Pytorch with CUDA later.
    
## Config

`src/settings/config.yml`

## Running

```
cd src
python app.py
```