# 📈 Stock Price Forecasting MLOps Project

This project demonstrates an **end-to-end MLOps workflow** for **stock market prediction** using **LSTM (Long Short-Term Memory networks)**, with automated retraining, deployment, and experiment tracking.

---

## 🚀 Features
- ✅ **Real stock data** from Yahoo Finance (`yfinance` API)
- ✅ **LSTM model** for next-day trend prediction (Up/Down)
- ✅ **MLflow integration** (experiment tracking, metrics, model versioning)
- ✅ **FastAPI** for serving predictions as an API
- ✅ **Docker containerization**
- ✅ **CI/CD with GitHub Actions** (automated retraining & redeployment weekly)
- ✅ **Unit testing** with `pytest`

---

## 📂 Project Structure
```
stock_mlops_project/
│── train.py               # LSTM model training + MLflow logging
│── app.py                 # FastAPI app serving predictions
│── requirements.txt       # Dependencies
│── Dockerfile             # Containerization setup
│── tests/
│    └── test_train.py     # Unit test for model sanity check
│── .github/
│    └── workflows/
│         └── ci-cd.yml    # CI/CD pipeline (weekly retraining)
```

---

## ⚙️ Setup Instructions

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/stock_mlops_project.git
cd stock_mlops_project
pip install -r requirements.txt
```

### 2. Train Model (with MLflow)
```bash
python train.py
mlflow ui
```
➡ Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to see experiment logs.

### 3. Run API Locally
```bash
uvicorn app:app --reload
```
➡ Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example request:
```
GET /predict?ticker=AAPL
```
Response:
```json
{
  "ticker": "AAPL",
  "prediction": "UP"
}
```

### 4. Dockerize
```bash
docker build -t stock-mlops-app .
docker run -p 8000:8000 stock-mlops-app
```
➡ API available at: `http://127.0.0.1:8000/docs`

### 5. Deploy to Render (Free Hosting)
1. Push repo to GitHub
2. Connect GitHub repo in [Render](https://render.com)
3. Deploy as **Web Service** (Docker)
4. Expose port `8000`
5. Access live API:  
   ```
   https://your-stock-mlops.onrender.com/docs
   ```

---

## 🔄 Automated Retraining (CI/CD)
GitHub Actions (`.github/workflows/ci-cd.yml`):
- Runs on **push** (CI/CD)
- Runs **weekly (cron job)** to retrain with the latest Yahoo Finance data
- Logs new metrics to MLflow
- Builds & pushes new Docker image to GitHub Container Registry (GHCR)
- Redeploys automatically (if auto-deploy enabled)

---

## 📊 Portfolio Value
This project demonstrates:
- **Deep Learning (LSTM)** for time-series forecasting
- **MLOps best practices** (CI/CD, Docker, MLflow)
- **Automated retraining** with real financial data
- **Scalable deployment** via Render or Google Cloud Run

Add to your resume/portfolio as:
> *“Built a stock price forecasting pipeline using LSTM and Yahoo Finance data. Implemented MLOps best practices with MLflow, Docker, GitHub Actions, and FastAPI for scalable deployment and automated retraining.”*

---

## 🏆 Next Steps / Extensions
- 🔹 Predict multiple tickers simultaneously
- 🔹 Extend prediction horizon (1 week ahead)
- 🔹 Add visualization dashboard (Plotly, Streamlit)
- 🔹 Deploy on **Google Cloud Run / AWS ECS**
- 🔹 Add monitoring (Prometheus + Grafana)
