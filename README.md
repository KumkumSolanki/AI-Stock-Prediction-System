# 📈 AI Stock Price Prediction System (LSTM + GRU + ARIMA)

This project is an end-to-end AI-based stock price forecasting system that predicts future prices, direction probability, and confidence intervals using deep learning and statistical models.

## 🚀 Features
- Real-time stock data using Yahoo Finance API
- Technical Indicators: SMA, EMA, RSI, MACD
- Models: LSTM, GRU, ARIMA
- 30-Day Forecast with Confidence Bands
- Up/Down Probability Prediction
- Interactive Web App using Streamlit

## 🧠 Model Comparison

| Model | RMSE |
|------|------|
| ARIMA | 25.19 |
| LSTM  | 4.82 |
| GRU   | 3.24 |

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Statsmodels
- Streamlit
- yFinance

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
