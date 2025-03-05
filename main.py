from src.data_fetch import fetch_yfinance_data, save_raw_data
from src.preprocessing import load_and_clean
from src.eda import plot_closing_price, plot_volatility, decompose_series
from src.models import train_arima
from src.forecasting import forecast_future, plot_forecast
from src.portfolio_optimization import optimize_portfolio, calculate_portfolio_metrics
import pandas as pd
def main():
    tickers = ["TSLA", "BND", "SPY"]
    
    # Task 1: Fetch and Preprocess Data
    print("Fetching data...")
    data = fetch_yfinance_data(tickers)
    save_raw_data(data)
    cleaned_data = load_and_clean(tickers)
    
    # Task 1: EDA
    for ticker in tickers:
        plot_closing_price(cleaned_data[ticker], ticker)
        plot_volatility(cleaned_data[ticker], ticker)
        decompose_series(cleaned_data[ticker], ticker)
    
    # Task 2: Train Model (ARIMA for TSLA)
    tsla_df = cleaned_data["TSLA"]
    model, predictions, test, metrics = train_arima(tsla_df)
    print("Model Metrics:", metrics)
    
    # Task 3: Forecast Future Trends
    forecast_df = forecast_future(model, tsla_df)
    plot_forecast(tsla_df, forecast_df, "TSLA")
    
    # Task 4: Portfolio Optimization
    portfolio_df = pd.concat([cleaned_data[ticker]['Close'] for ticker in tickers], axis=1)
    portfolio_df.columns = tickers
    optimal_weights = optimize_portfolio(portfolio_df)
    ret, vol, sharpe = calculate_portfolio_metrics(portfolio_df, optimal_weights)
    print(f"Optimal Weights: {dict(zip(tickers, optimal_weights))}")
    print(f"Portfolio Return: {ret:.4f}, Volatility: {vol:.4f}, Sharpe Ratio: {sharpe:.4f}")

if __name__ == "__main__":
    main()