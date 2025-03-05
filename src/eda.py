# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.seasonal import seasonal_decompose
# import os
# print("Current Working Directory:", os.getcwd())
# def plot_closing_price(df, ticker):
#     """Plot closing price over time."""
#     plt.figure(figsize=(10, 6))
#     plt.plot(df.index, df['Close'], label=f'{ticker} Close Price')
#     plt.title(f'{ticker} Closing Price (2015-2025)')
#     plt.xlabel('Date')
#     plt.ylabel('Price (USD)')
#     plt.legend()
#     plt.savefig(f'../reports/{ticker}_closing_price.png')
#     plt.close()

# def plot_volatility(df, ticker, window=30):
#     """Plot rolling mean and standard deviation for volatility."""
#     rolling_mean = df['Close'].rolling(window=window).mean()
#     rolling_std = df['Close'].rolling(window=window).std()
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(df.index, rolling_mean, label='Rolling Mean')
#     plt.plot(df.index, rolling_std, label='Rolling Std Dev')
#     plt.title(f'{ticker} Volatility (Rolling Window = {window})')
#     plt.legend()
#     plt.savefig(f'../reports/{ticker}_volatility.png')
#     plt.close()

# def decompose_series(df, ticker):
#     """Decompose time series into trend, seasonal, and residual components."""
#     decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)  # Yearly seasonality
#     decomposition.plot()
#     plt.savefig(f'../reports/{ticker}_decomposition.png')
#     plt.close()

# if __name__ == "__main__":
#     tickers = ["TSLA", "BND", "SPY"]
#     for ticker in tickers:
#         df = pd.read_csv(f"../data/processed/{ticker}_processed.csv", index_col='Date', parse_dates=True)
#         plot_closing_price(df, ticker)
#         plot_volatility(df, ticker)
#         decompose_series(df, ticker)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# Set the project root as the working directory (optional, ensures consistency)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
print(f"Working Directory Set To: {os.getcwd()}")

def plot_closing_price(df, ticker):
    """Plot closing price over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} Closing Price (2015-2025)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    os.makedirs('reports', exist_ok=True)  # Create reports/ if it doesn’t exist
    try:
        plt.savefig(f'reports/{ticker}_closing_price.png')
        print(f"Saved plot: reports/{ticker}_closing_price.png")
    except Exception as e:
        print(f"Error saving {ticker}_closing_price.png: {e}")
    plt.close()

def plot_volatility(df, ticker, window=30):
    """Plot rolling mean and standard deviation for volatility."""
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, rolling_mean, label='Rolling Mean')
    plt.plot(df.index, rolling_std, label='Rolling Std Dev')
    plt.title(f'{ticker} Volatility (Rolling Window = {window})')
    plt.legend()
    os.makedirs('reports', exist_ok=True)  # Create reports/ if it doesn’t exist
    try:
        plt.savefig(f'reports/{ticker}_volatility.png')
        print(f"Saved plot: reports/{ticker}_volatility.png")
    except Exception as e:
        print(f"Error saving {ticker}_volatility.png: {e}")
    plt.close()

def decompose_series(df, ticker):
    """Decompose time series into trend, seasonal, and residual components."""
    decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)
    decomposition.plot()
    os.makedirs('reports', exist_ok=True)  # Create reports/ if it doesn’t exist
    try:
        plt.savefig(f'reports/{ticker}_decomposition.png')
        print(f"Saved plot: reports/{ticker}_decomposition.png")
    except Exception as e:
        print(f"Error saving {ticker}_decomposition.png: {e}")
    plt.close()

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    for ticker in tickers:
        df = pd.read_csv(f"data/processed/{ticker}_processed.csv", index_col='Date', parse_dates=True)
        plot_closing_price(df, ticker)
        plot_volatility(df, ticker)
        decompose_series(df, ticker)