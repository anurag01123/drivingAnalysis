from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from tsai.all import *
from tsai.inference import load_learner


app = FastAPI()

class Cord(BaseModel):
    timestamp: str
    latitude: str
    longitude: str
    speed: Optional[str] = ...
    
# class JsonList(BaseModel):
#     data: List[Cord]

origins = ['https://localhost:3000',
"http://localhost",
"http://localhost:5500",
"http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return("hello there!")

collected_data = []    
@app.post("/getClass")
async def post_todo(dataRec:List[Cord]):
    for item in dataRec:
        collected_data.append([item.timestamp, item.latitude, item.longitude, item.speed])
    print(collected_data)
    df = pd.DataFrame(collected_data, columns =['timestamp', 'latitude', 'longitude', 'speed'])
    df.columns = ["time","Latitude","Longitude","speed"]
    datalimit = 1630
    df["TripID"] = 'T-0'
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors ='ignore')
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors ='ignore')
    df["speed"] = pd.to_numeric(df["speed"], errors ='ignore')
    df = df.sort_values(by=['time'], ascending=True)
    if(len(df)>datalimit):
        df = df[:datalimit]
    else:
        dframe2 = np.zeros((datalimit-len(df), 5))
        dframe2 = pd.DataFrame(dframe2, columns=["time","Latitude", "Longitude", "speed", "TripID"])
        dframe2["TripID"] = 'T-0'
        newFrame = [df, dframe2]
        df = pd.concat(newFrame)
    # getting test data
    X_test, y_test = df2Xy(df, sample_col='TripID', sort_by ='time',data_cols=['Latitude','Longitude','speed'] ,steps_in_rows=True)
    clf = load_learner("/home/dbx/finalApp/backend/trainedModel.pkl")
    probas, target, preds = clf.get_X_preds(X_test)
    if(preds=='[0.0]'):
        return{"Class":"Safe"}
    else:
        return{"Class":"Unsafe"}

# @app.get("/score")
# def getScore():
#     dframe = pd.read_csv('/home/dbx/finalApp/download_bike_speed.csv', header=None)
#     dframe.columns = ["time","Latitude","Longitude","speed"]
#     dframe["TripID"] = 'T-0'
#     dframe["Latitude"] = pd.to_numeric(dframe["Latitude"], errors ='ignore')
#     dframe["Longitude"] = pd.to_numeric(dframe["Longitude"], errors ='ignore')
#     dframe["speed"] = pd.to_numeric(dframe["speed"], errors ='ignore')
#     dframe = dframe.sort_values(by=['time'], ascending=True)
#     dframe2 = np.zeros((1630-len(dframe), 5))
#     dframe2 = pd.DataFrame(dframe2, columns=["time","Latitude", "Longitude", "speed", "TripID"])
#     dframe2["TripID"] = 'T-0'
#     newFrame = [dframe, dframe2]
#     dframe = pd.concat(newFrame)
#     # getting test data
#     X_test, y_test = df2Xy(dframe, sample_col='TripID', sort_by ='time',data_cols=['Latitude','Longitude','speed'] ,steps_in_rows=True)
#     clf = load_learner("/home/dbx/finalApp/backend/trainedModel.pkl")
#     probas, target, preds = clf.get_X_preds(X_test)
#     if(preds=='[0.0]'):
#         return{"Class":"Safe"}
#     else:
#         return{"Class":"Unsafe"}