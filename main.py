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
    
origins = ["*"]

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

    
@app.post("/getClass")
def class_fetch(dataRec:List[Cord]):
    collected_data = []
    for item in dataRec:
        collected_data.append([item.timestamp, item.latitude, item.longitude, item.speed])
    print(collected_data)
    df = pd.DataFrame(collected_data, columns =['timestamp', 'latitude', 'longitude', 'speed'])
    df.columns = ["time","Latitude","Longitude","speed"]
    datalimit = 2000
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
    clf = load_learner("trainedModel.pkl")
    probas, target, preds = clf.get_X_preds(X_test)
    print(preds)
    if(preds=='[0.0]'):
        df = df.iloc[0:0]
        collected_data = []
        return{"Class":"Safe"}
    else:
        df = df.iloc[0:0]
        collected_data = []
        return{"Class":"Unsafe"}
