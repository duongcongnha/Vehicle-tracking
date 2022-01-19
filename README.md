# Vehicle tracking with Yolov5 + Deep Sort with PyTorch




## Before running the tracker


`pip install -r requirements.txt`

   
## Config

`src/settings/config.yml`

## Running

```
cd application\main
python app_track.py
```

## FastAPI

```
cd application\main
uvicorn app_API:app --host 0.0.0.0 --port 8000 --reload
```
