import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

st.markdown("""
<style>
.big-title {
    font-size:36px !important;
    font-weight:700;
}
.download-btn button {
    background-color: #4CAF50;
    color: white;
    padding: 0.6em 1.5em;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}
.download-btn button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">Cryptocurrency Forecasting Dashboard</p>', unsafe_allow_html=True)

days = st.slider("Select Forecast Horizon (Days)", 30, 120, 60)

btc = yf.download("BTC-USD", period="5y")

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

btc = btc.reset_index()

st.subheader("Bitcoin Closing Price")
st.line_chart(btc.set_index("Date")["Close"])

btc["MA_7"] = btc["Close"].rolling(7).mean()
btc["MA_30"] = btc["Close"].rolling(30).mean()

st.subheader("Moving Averages")
st.line_chart(btc.set_index("Date")[["Close", "MA_7", "MA_30"]])

st.markdown('<div class="download-btn">', unsafe_allow_html=True)
csv = btc.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download BTC 5-Year Data",
    data=csv,
    file_name="btc_5year_data.csv",
    mime="text/csv",
)
st.markdown('</div>', unsafe_allow_html=True)

btc_ts = btc.set_index("Date")["Close"]

train = btc_ts[:-days]
test = btc_ts[-days:]

model_option = st.selectbox("Select Model", ["ARIMA", "Prophet", "XGBoost"])

if model_option == "ARIMA":
    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    plt.figure(figsize=(12,6))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(test.index, forecast.values, label="ARIMA Forecast")
    plt.legend()
    st.pyplot(plt)

elif model_option == "Prophet":
    df = btc[["Date", "Close"]].copy()
    df.columns = ["ds", "y"]
    
    train_p = df[:-days]
    test_p = df[-days:]
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(train_p)
    
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    forecast_test = forecast.set_index("ds").loc[test_p["ds"]]
    rmse = np.sqrt(mean_squared_error(test_p["y"], forecast_test["yhat"]))
    
    plt.figure(figsize=(12,6))
    plt.plot(test_p["ds"], test_p["y"], label="Actual")
    plt.plot(test_p["ds"], forecast_test["yhat"], label="Prophet Forecast")
    plt.legend()
    st.pyplot(plt)

elif model_option == "XGBoost":
    btc["lag_1"] = btc["Close"].shift(1)
    btc["lag_7"] = btc["Close"].shift(7)
    btc["rolling_mean_7"] = btc["Close"].rolling(7).mean()
    btc["rolling_std_7"] = btc["Close"].rolling(7).std()
    
    btc_ml = btc.dropna()
    
    features = ["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]
    
    X = btc_ml[features]
    y = btc_ml["Close"]
    
    X_train, X_test = X[:-days], X[-days:]
    y_train, y_test = y[:-days], y[-days:]
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, y_pred, label="XGBoost Forecast")
    plt.legend()
    st.pyplot(plt)

st.subheader("Model RMSE")
st.write(rmse)
