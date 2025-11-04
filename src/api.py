from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from infer import predict_anomaly

ANOMALY_THRESHOLD = 0.5  # Replace ewith computed threshold

app = FastAPI(title='Autoencoder Anomaly Detection API')


# Define request schema
class InputData(BaseModel):
    features: List[float]


class BatchInput(BaseModel):
    data: List[List[float]]


# Single prediction
@app.post('/predict')
def predict_single(sample: InputData):
    arr = np.array(sample.features).reshape(1, -1)
    result = predict_anomaly(arr, ANOMALY_THRESHOLD)
    return result[0]


# Batch prediction
@app.post('/predict_batch')
def predict_batch(batch: BatchInput):
    arr = np.array(batch.data)
    results = predict_anomaly(arr, ANOMALY_THRESHOLD)
    return results
