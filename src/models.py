import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_arima(df, train_split=0.8):
    """Train an ARIMA model using auto_arima."""
    train_size = int(len(df) * train_split)
    train, test = df['Close'][:train_size], df['Close'][train_size:]
    
    model = auto_arima(train, seasonal=False, stepwise=True, trace=True)
    fitted_model = model.fit(train)
    predictions = fitted_model.predict(n_periods=len(test))
    
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
    model, predictions, test, metrics = train_arima(df)
    print("ARIMA Metrics:", metrics)