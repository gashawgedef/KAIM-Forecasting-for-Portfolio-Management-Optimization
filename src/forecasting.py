# import pandas as pd
# from pmdarima import auto_arima
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import numpy as np
# import matplotlib.pyplot as plt

# def train_arima_model(df, train_size=0.8):
#     """Train ARIMA model and forecast."""
#     train_len = int(len(df) * train_size)
#     train, test = df['Close'][:train_len], df['Close'][train_len:]
    
#     # Fit ARIMA model
#     model = auto_arima(train, seasonal=False, trace=True)
#     forecast = model.predict(n_periods=len(test))
    
#     # Evaluation
#     mae = mean_absolute_error(test, forecast)
#     rmse = np.sqrt(mean_squared_error(test, forecast))
#     print(f"MAE: {mae}, RMSE: {rmse}")
    
#     return model, forecast, test

# def plot_forecast(train, test, forecast, ticker):
#     """Plot forecast vs actual."""
#     plt.figure(figsize=(10, 6))
#     plt.plot(train, label='Training Data')
#     plt.plot(test.index, test, label='Actual')
#     plt.plot(test.index, forecast, label='Forecast')
#     plt.legend()
#     plt.title(f'{ticker} Price Forecast')
#     plt.savefig(f'../reports/final_submission/{ticker}_forecast.png')
#     plt.close()

# if __name__ == "__main__":
#     ticker = "TSLA"
#     df = pd.read_csv(f"../data/processed/{ticker}_processed.csv", index_col='Date')
#     model, forecast, test = train_arima_model(df)
#     plot_forecast(df['Close'][:len(df)-len(test)], test, forecast, ticker)
    
#     # Forecast 6 months ahead (assume 126 trading days)
#     future_forecast = model.predict(n_periods=126)
import pandas as pd
import matplotlib.pyplot as plt

from models import train_arima

def forecast_future(model, df, steps=252):  # 252 trading days ~ 1 year
    """Generate future forecasts using the trained model."""
    forecast = model.predict(n_periods=steps)
    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='B')[1:]
    
    forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Forecast'])
    return forecast_df

def plot_forecast(df, forecast_df, ticker):
    """Plot historical data with forecast."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Historical')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='orange')
    plt.title(f'{ticker} Stock Price Forecast')
    plt.legend()
    plt.savefig(f'reports/{ticker}_forecast.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/TSLA_processed.csv", index_col='Date', parse_dates=True)
    model, _, _, _ = train_arima(df)  # From models.py
    forecast_df = forecast_future(model, df)
    plot_forecast(df, forecast_df, "TSLA")