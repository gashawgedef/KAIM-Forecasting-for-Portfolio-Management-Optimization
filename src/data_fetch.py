# import yfinance as yf
# import pandas as pd
# import os

# def fetch_yfinance_data(ticker, start_date, end_date, save_path):
#     """Fetch historical data from YFinance and save to CSV."""
#     stock = yf.Ticker(ticker)
#     df = stock.history(start=start_date, end=end_date)
#     df.to_csv(save_path)
#     return df

# if __name__ == "__main__":
#     tickers = ["TSLA", "BND", "SPY"]
#     start_date = "2015-01-01"
#     end_date = "2025-01-31"
#     output_dir = "../data/raw/"
    
#     os.makedirs(output_dir, exist_ok=True)
#     for ticker in tickers:
#         save_path = os.path.join(output_dir, f"{ticker}_2015_2025.csv")
#         df = fetch_yfinance_data(ticker, start_date, end_date, save_path)
#         print(f"Data for {ticker} fetched and saved to {save_path}")

import yfinance as yf
import pandas as pd
import os

def fetch_yfinance_data(tickers, start_date="2015-01-01", end_date="2025-01-31"):
    """Fetch historical data from YFinance for given tickers."""
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        data[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def save_raw_data(data, output_dir="data/raw"):
    """Save raw data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    for ticker, df in data.items():
        df.to_csv(f"{output_dir}/{ticker}_raw.csv")
    print(f"Raw data saved to {output_dir}")

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    data = fetch_yfinance_data(tickers)
    save_raw_data(data)