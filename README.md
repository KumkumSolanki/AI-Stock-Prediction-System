# 📈 AI Stock Price Prediction System (LSTM + GRU + ARIMA)

This project is an end-to-end AI-based stock price forecasting system that predicts future prices, direction probability, and confidence intervals using deep learning and statistical models.

## 🚀 Features
- Real-time stock data using Yahoo Finance API
- Technical Indicators: SMA, EMA, RSI, MACD
- Models: LSTM, GRU, ARIMA
- 30-Day Forecast with Confidence Bands
- Up/Down Probability Prediction
- Interactive Web App using Streamlit
## 🎯 Project Motivation
Stock markets are highly volatile and nonlinear. Traditional statistical models often fail to capture long-term dependencies and complex patterns. This project aims to leverage deep learning architectures (LSTM, GRU) combined with technical indicators to improve forecasting accuracy and provide uncertainty-aware predictions for better decision making.

## 🧩 Problem Statement
Given historical stock prices and technical indicators, predict:
1. Future stock prices for the next 30 days  
2. Probability of upward or downward movement  
3. Confidence interval for predictions  
4. Compare deep learning models with classical ARIMA

## ⚙️ System Pipeline
1. Data Collection using Yahoo Finance API  
2. Feature Engineering (SMA, EMA, RSI, MACD)  
3. Data Scaling using MinMaxScaler  
4. Sequence Generation (60-day window)  
5. Model Training (LSTM, GRU, ARIMA)  
6. Model Evaluation (RMSE)  
7. Forecasting with Confidence Bands  
8. Deployment using Streamlit Web App  

## 📊 Results
- GRU achieved the lowest RMSE and best generalization.
- The system successfully predicts trend direction with probability scores.
- Confidence bands help visualize uncertainty in volatile markets.

## 🔮 Future Improvements
- Add Transformer-based time series model  
- Integrate financial news sentiment analysis  
- Multi-stock portfolio forecasting  
- Reinforcement learning for trading strategy  
- Cloud deployment with CI/CD  

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

## 🧑‍💻 Author
Kumkum Solanki  
B.Tech Final Year | AI & Machine Learning  

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

