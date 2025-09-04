from fastapi import FastAPI
import mlflow
import mlflow.pytorch
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import csv
from pathlib import Path

app = FastAPI()

model = mlflow.pytorch.load_model("runs:/<replace_with_run_id>/model")
model.eval()

DATA_LOG = Path("logged_predictions.csv")

if not DATA_LOG.exists():
    with open(DATA_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ticker", "prediction"])

def preprocess(ticker="AAPL", window=60):
    df = yf.download(ticker, period="90d")
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X = scaled[-window:]
    return torch.tensor(X, dtype=torch.float32).unsqueeze(0)

@app.get("/predict")
def predict(ticker: str = "AAPL"):
    X = preprocess(ticker)
    with torch.no_grad():
        output = model(X)
        pred = torch.argmax(output, axis=1).item()

    prediction = "UP" if pred == 1 else "DOWN"
    timestamp = int(time.time())

    with mlflow.start_run(run_name="inference-logging", nested=True):
        mlflow.log_param("ticker", ticker)
        mlflow.log_metric("prediction", pred)
        mlflow.log_metric("timestamp", timestamp)

    with open(DATA_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, ticker, prediction])

    return {"ticker": ticker, "prediction": prediction, "logged": True}
