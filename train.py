import mlflow
import mlflow.pytorch
import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h[:, -1, :])
        return out

def load_yahoo_data(ticker="AAPL", window=60):
    df = yf.download(ticker, period="2y")
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window, len(scaled)-1):
        X.append(scaled[i-window:i])
        y.append(1 if scaled[i+1, 3] > scaled[i, 3] else 0)  # Up/Down

    return np.array(X), np.array(y), scaler

def merge_logged_data(X, y, window=60):
    log_file = Path("logged_predictions.csv")
    if log_file.exists() and log_file.stat().st_size > 0:
        logged = pd.read_csv(log_file)
        if not logged.empty:
            for _, row in logged.iterrows():
                dummy_features = np.zeros((window, 5))  # placeholder features
                label = 1 if row["prediction"] == "UP" else 0
                X = np.vstack([X, [dummy_features]])
                y = np.append(y, label)
    return X, y

def train_model(ticker="AAPL", epochs=5, batch_size=32, lr=0.001, window=60):
    X, y, scaler = load_yahoo_data(ticker, window=window)
    X, y = merge_logged_data(X, y, window=window)

    X_train, y_train = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StockLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_train)
        preds = torch.argmax(preds, axis=1).numpy()
        acc = accuracy_score(y, preds)

    return model, acc, scaler

if __name__ == "__main__":
    mlflow.set_experiment("stock-lstm")

    with mlflow.start_run():
        model, acc, scaler = train_model()

        mlflow.log_metric("accuracy", acc)
        mlflow.pytorch.log_model(model, "model")

        print(f"Training Accuracy: {acc:.4f}")
