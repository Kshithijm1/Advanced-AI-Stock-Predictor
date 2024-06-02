import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function to fetch stock data using yfinance
def fetch_stock_data(ticker, period='2y', interval='1d'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.dropna(inplace=True)  # Remove missing values
    if len(df) < 60:
        raise ValueError("Not enough data fetched. Ensure the period and interval result in at least 60 data points.")
    return df

# Function to compute the Relative Strength Index (RSI)
def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute Bollinger Bands
def compute_bollinger_bands(series, window=20, num_std_dev=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# Function to compute the Stochastic Oscillator
def compute_stochastic_oscillator(df, period=14):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    stoch = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return stoch

# Function to compute the Average True Range (ATR)
def compute_ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr

# Function to compute the Volume Weighted Average Price (VWAP)
def compute_VWAP(df):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

# Function to add various technical indicators to the stock data
def add_technical_indicators(df):
    df['RSI'] = compute_RSI(df['Close'], 14)
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['Close'])
    df['Stochastic_Oscillator'] = compute_stochastic_oscillator(df)
    df['ATR'] = compute_ATR(df)
    df['VWAP'] = compute_VWAP(df)
    return df

# Function to preprocess data for model input
def preprocess_data(df):
    scaler = MinMaxScaler()
    df = df.dropna()  # Remove rows with NaN values
    data = df.values
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 3])  # Assuming 'Close' is the target variable

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler
