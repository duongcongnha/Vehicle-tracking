from fastapi import FastAPI, Form 
from typing import Optional
from datetime import datetime, time, timedelta
from infrastructure.database.common import get_vehicles_in_timerange


app = FastAPI()


@app.get("/") 
async def root():
    return {"message": "Hello World"}

@app.get("/sum-2-numbers")
async def sum(a, b):
    return {f"sum of {a} and {b}": (int(a) + int(b))}


@app.post("/get-vehicles-in-interval")
async def get_vehicles_in_interval(start_datetime: str = Form(default=(datetime.now() - timedelta(days=1)).strftime("%d-%m-%Y %H:%M:%S")),
                        end_datetime: str = Form(default=datetime.now().strftime("%d-%m-%Y %H:%M:%S"))):

    # "12/11/2018 09:15:32"
    FMT = "%d-%m-%Y %H:%M:%S"

    start_datetime = datetime.strptime(start_datetime, FMT)
    end_datetime = datetime.strptime(end_datetime, FMT)
    
    all_vehicles = get_vehicles_in_timerange(start_datetime, end_datetime)
    
    return {"all_vehicles": all_vehicles}

# @app.post("/login/")
# async def login(username: str = Form(default="nhap username do"), password: str = Form(...)):
#     return {"username": username}


# uvicorn app_API:app --host 0.0.0.0 --port 8000 --reload
