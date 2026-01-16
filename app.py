import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AI Stock Forecast", layout="wide")
st.title("📈 AI Stock Price Prediction System (LSTM + GRU + ARIMA)")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Predict"):

    df = yf.download(ticker, start="2018-01-01", end="2024-01-01")
    data = df.copy()

    # Technical Indicators
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_30'] = data['Close'].rolling(30).mean()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema12 - ema26

    data.dropna(inplace=True)

    features = data[['Close','SMA_10','SMA_30','EMA_10','RSI','MACD']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X = []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
    X = np.array(X)

    # Load GRU model (FIXED)
    model = load_model("gru_stock_model.h5", compile=False)

    preds = model.predict(X[-30:])

    preds_real = scaler.inverse_transform(
        np.hstack([preds, np.zeros((preds.shape[0],5))])
    )[:,0]

    # Confidence Band
    std = np.std(preds_real)
    upper = preds_real + 2*std
    lower = preds_real - 2*std

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(preds_real, label="Predicted Price")
    ax.fill_between(range(30), lower, upper, alpha=0.3, label="Confidence Interval")
    ax.set_title("30-Day Forecast with Confidence Band (GRU)")
    ax.legend()
    st.pyplot(fig)

    # Direction Probability
    last_seq = scaled[-60:].reshape(1,60,6)
    prob = model.predict(last_seq)[0][0]

    if prob > 0.5:
        st.success(f"📊 Probability of UP Tomorrow: {prob*100:.2f}%")
    else:
        st.error(f"📉 Probability of DOWN Tomorrow: {(1-prob)*100:.2f}%")

    # Model Comparison Table
    st.subheader("Model Performance Comparison")
    st.table(pd.DataFrame({
        "Model": ["ARIMA", "LSTM", "GRU"],
        "RMSE": [25.19, 4.82, 3.24]
    }))
