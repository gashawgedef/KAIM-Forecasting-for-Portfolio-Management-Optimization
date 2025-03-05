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
# import pandas as pd
# import matplotlib.pyplot as plt

# from models import train_arima

# def forecast_future(model, df, steps=252):  # 252 trading days ~ 1 year
#     """Generate future forecasts using the trained model."""
#     forecast = model.predict(n_periods=steps)
#     last_date = pd.to_datetime(df.index[-1])
#     future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='B')[1:]
    
#     forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Forecast'])
#     return forecast_df

# def plot_forecast(df, forecast_df, ticker):
#     """Plot historical data with forecast."""
#     plt.figure(figsize=(12, 6))
#     plt.plot(df.index, df['Close'], label='Historical')
#     plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='orange')
#     plt.title(f'{ticker} Stock Price Forecast')
#     plt.legend()
#     plt.savefig(f'../reports/{ticker}_forecast.png')
#     plt.close()

# if __name__ == "__main__":
#     df = pd.read_csv("../data/processed/TSLA_processed.csv", index_col='Date', parse_dates=True)
#     model, _, _, _ = train_arima(df)  # From models.py
#     forecast_df = forecast_future(model, df)
#     plot_forecast(df, forecast_df, "TSLA")

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress warnings

def train_arima(df, train_split=0.8):
    """Train an ARIMA model using auto_arima."""
    train_size = int(len(df) * train_split)
    train, test = df['Close'][:train_size], df['Close'][train_size:]
    
    # Check and clean input data
    print(f"Train NaN count: {train.isna().sum()}")
    print(f"Test NaN count: {test.isna().sum()}")
    if train.isna().any() or test.isna().any():
        print("Warning: Input data contains NaN. Cleaning...")
        train = train.dropna()
        test = test.dropna()
        print(f"After cleaning: Train size={len(train)}, Test size={len(test)}")
    
    # Train model
    model = auto_arima(train, seasonal=False, stepwise=True, trace=True)
    fitted_model = model.fit(train)
    predictions = fitted_model.predict(n_periods=len(test))
    
    # Check predictions for NaN
    print(f"Predictions NaN count: {np.isnan(predictions).sum()}")
    if np.isnan(predictions).any():
        print("Warning: Predictions contain NaN. Replacing with last train value...")
        predictions = np.nan_to_num(predictions, nan=train.iloc[-1])
    
    # Align predictions with test index
    predictions = pd.Series(predictions, index=test.index)
    
    # Final NaN check before metrics
    print(f"Test NaN count after alignment: {test.isna().sum()}")
    print(f"Predictions NaN count after alignment: {predictions.isna().sum()}")
    
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    return fitted_model, predictions, test, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def train_sarima(df, train_split=0.8):
    """Train a SARIMA model."""
    train_size = int(len(df) * train_split)
    train, test = df['Close'][:train_size], df['Close'][train_size:]
    
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 252))
    fitted_model = model.fit(disp=False)
    predictions = fitted_model.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    return fitted_model, predictions, test, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

if __name__ == "__main__":
    df = pd.read_csv("data/processed/TSLA_processed.csv", index_col='Date', parse_dates=True)
    print("Initial Data Check:")
    print(f"Close NaN count: {df['Close'].isna().sum()}")
    
    model, predictions, test, metrics = train_arima(df)
    print("ARIMA Metrics:", metrics)