"""
Crypto-Analysis-Tool
====================

Author: Emmanuel Jean-Louis Wojcik
Date: 2024-12-31
Version: 1.0
Description: 
    This script analyzes Bitcoin price data, calculates technical indicators, 
    generates trading signals, and predicts future prices.

Dependencies:
    - pandas
    - numpy
    - yfinance
    - matplotlib
    - pmdarima
    - sklearn

License: MIT
"""

# Import necessary libraries for data manipulation, visualization, and time series analysis
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import pmdarima as pm
import time

# Setup logging configuration for error tracking and information logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_bitcoin_data(retries=3, delay=5):
    # Attempt to fetch Bitcoin data with error handling and retry mechanism
    for attempt in range(retries):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 90 days for more data points
            btc_data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1h')
            if btc_data.empty:
                raise ValueError("No data fetched from yfinance")
            btc_data = btc_data.asfreq('h')  # Ensure hourly frequency
            return btc_data
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to fetch Bitcoin data: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before next attempt
            else:
                logging.error("All retries failed. Exiting.")
                return None


def validate_data(df):
    # Validate the fetched data for completeness and consistency
    if df is None:
        return False
    if df.isna().any().any():
        df = df.fillna(method='ffill')  # Forward fill NaN values
        logging.warning("Data contained NaN values, which have been filled.")
    if not all(df.index.to_series().diff().dropna() == timedelta(hours=1)):
        logging.warning("Data does not have consistent hourly intervals")
        return False
    return True


def calculate_technical_indicators(df):
    # Compute various technical indicators for financial analysis
    if df is None:
        return None

    result = pd.DataFrame(index=df.index)
    for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
        result[col] = df[col]

    # Calculate Moving Averages
    periods = [20, 50, 100]
    for period in periods:
        result[f'SMA_{period}'] = result['Close'].rolling(window=period).mean()
        result[f'EMA_{period}'] = result['Close'].ewm(span=period, adjust=False).mean()

    # Calculate Moving Average Convergence Divergence (MACD)
    result['MACD'] = result['Close'].ewm(span=12, adjust=False).mean() - result['Close'].ewm(span=26, adjust=False).mean()
    result['Signal_Line'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_Hist'] = result['MACD'] - result['Signal_Line']

    # Calculate Relative Strength Index (RSI)
    delta = result['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    result['RSI'] = 100 - (100 / (1 + avg_gain / avg_loss))

    # Add time-based indicators
    result['Hour'] = result.index.hour
    result['DayOfWeek'] = result.index.dayofweek

    # Calculate Volatility and Moving Average of Volume
    result['Volatility'] = result['Close'].rolling(window=20).std()
    result['Volume_MA'] = result['Volume'].rolling(window=20).mean()

    # Generate trading signals
    result['Buy_Signal'] = generate_buy_signals(result)
    result['Sell_Signal'] = generate_sell_signals(result)

    return result


def generate_buy_signals(df):
    # Criteria for generating buy signals:
    # - RSI below 30 (oversold)
    # - MACD above Signal Line (bullish crossover)
    # - Price above 20-day SMA (indicating potential upward trend)
    conditions = (
            (df['RSI'] < 30) &
            (df['MACD'] > df['Signal_Line']) &
            (df['Close'] > df['SMA_20'])
    )
    return conditions.astype(int)


def generate_sell_signals(df):
    # Criteria for generating sell signals:
    # - RSI above 70 (overbought)
    # - MACD below Signal Line (bearish crossover)
    # - Price below 20-day SMA (indicating potential downward trend)
    conditions = (
            (df['RSI'] > 70) &
            (df['MACD'] < df['Signal_Line']) &
            (df['Close'] < df['SMA_20'])
    )
    return conditions.astype(int)


def plot_analysis(df):
    # Plot the technical analysis including price, moving averages, and signals
    if df is None:
        logging.error("No data to plot")
        return

    fig = plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['Close'], label='Price', color='blue')
    ax1.plot(df.index, df['SMA_20'], label='20 MA', color='orange')

    buy_points = df[df['Buy_Signal'] == 1]
    ax1.scatter(buy_points.index, buy_points['Close'], color='green', marker='^', label='Buy Signal')

    sell_points = df[df['Sell_Signal'] == 1]
    ax1.scatter(sell_points.index, sell_points['Close'], color='red', marker='v', label='Sell Signal')

    ax1.set_title('Bitcoin Price with Trading Signals')
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_title('RSI Indicator')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def analyze_timing(df):
    # Analyze best times for trading based on historical data
    if df is None:
        logging.error("No data for timing analysis")
        return {}

    timing_analysis = {
        'Best_Trading_Hours': df.groupby('Hour')['Volatility'].mean().nlargest(3).index.tolist(),
        'Best_Trading_Days': df.groupby('DayOfWeek')['Volume'].mean().nlargest(3).index.tolist(),
        'High_Volume_Times': df[df['Volume'] > df['Volume_MA']].groupby('Hour').size().nlargest(3).index.tolist()
    }

    days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    timing_analysis['Best_Trading_Days'] = [days_map[day] for day in timing_analysis['Best_Trading_Days']]

    return timing_analysis


def predict_price(df, steps=10):
    # Use ARIMA model to predict future prices
    if df is None:
        logging.error("No data for prediction")
        return None

    df_diff = df['Close'].diff().dropna()

    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df_diff)
    if result[1] > 0.05:
        logging.warning("Series might not be stationary. Consider further differencing or transformations.")

    try:
        # Use auto_arima for automatic model selection
        model = pm.auto_arima(df['Close'], start_p=1, start_q=1,
                              test='adf',
                              max_p=3, max_q=3,
                              m=24,  # 24 for daily seasonality in hourly data
                              d=None,
                              seasonal=True,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        fitted_model = model.fit(df['Close'])
        forecast = fitted_model.predict(n_periods=steps)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps, freq='h')
        prediction_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicted_Close'])

        return prediction_df
    except Exception as e:
        logging.error(f"Error in ARIMA prediction: {e}")
        return None


def plot_predictions(df, predictions):
    # Visualize historical price data alongside the predictions
    if df is None or predictions is None:
        logging.error("No data to plot for predictions")
        return

    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'], label='Historical Price', color='blue')
    plt.plot(predictions.index, predictions['Predicted_Close'], label='Predicted Price', color='red', linestyle='--')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main():
    # Main function to orchestrate data fetching, analysis, and predictions
    btc_data = fetch_bitcoin_data()
    if btc_data is None or not validate_data(btc_data):
        logging.error("Data fetching or validation failed. Exiting.")
        return

    df_analyzed = calculate_technical_indicators(btc_data)

    if df_analyzed is not None:
        timing = analyze_timing(df_analyzed)

        logging.info("\nTrading Timing Analysis:")
        logging.info(f"Best Trading Hours (UTC): {timing['Best_Trading_Hours']}")
        logging.info(f"Best Trading Days: {timing['Best_Trading_Days']}")
        logging.info(f"High Volume Hours (UTC): {timing['High_Volume_Times']}")

        latest = df_analyzed.iloc[-1]
        logging.info("\nCurrent Trading Signals:")
        logging.info(f"Price: ${latest['Close']:.2f}")
        logging.info(f"RSI: {latest['RSI']:.2f}")
        logging.info(f"MACD: {latest['MACD']:.2f}")

        if latest['Buy_Signal']:
            logging.info("STRONG BUY SIGNAL")
        elif latest['Sell_Signal']:
            logging.info("STRONG SELL SIGNAL")
        else:
            logging.info("No clear signal - HOLD")

        plot_analysis(df_analyzed)

        predictions = predict_price(df_analyzed, steps=10)
        if predictions is not None:
            logging.info("\nBitcoin Price Prediction for next 10 intervals:")
            logging.info(predictions)
            plot_predictions(df_analyzed, predictions)
    else:
        logging.error("Failed to calculate technical indicators.")


if __name__ == "__main__":
    main()
