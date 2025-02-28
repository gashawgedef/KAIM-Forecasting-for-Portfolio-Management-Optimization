import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def preprocess_data(df):
    """Clean and preprocess financial data."""
    # Ensure correct data types
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    # Handle missing values
    df = df.interpolate(method='linear').fillna(method='bfill')
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def plot_eda(df, ticker):
    """Perform EDA and save plots."""
    # Closing price plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} Closing Price Over Time')
    plt.savefig(f'../reports/interim_submission/{ticker}_close.png')
    plt.close()
    
    # Decomposition
    decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)
    decomposition.plot()
    plt.savefig(f'../reports/interim_submission/{ticker}_decomposition.png')
    plt.close()

if __name__ == "__main__":
    ticker = "TSLA"
    df = pd.read_csv(f"../data/raw/{ticker}_2015_2025.csv", index_col='Date')
    df = preprocess_data(df)
    df.to_csv(f"../data/processed/{ticker}_processed.csv")
    plot_eda(df, ticker)