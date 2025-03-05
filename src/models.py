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
# import pandas as pd
# from pmdarima import auto_arima
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import numpy as np
# import warnings
# import matplotlib.pyplot as plt
# import os

# warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress warnings

# def train_arima(df, train_split=0.8):
#     """Train an ARIMA model using auto_arima."""
#     train_size = int(len(df) * train_split)
#     train, test = df['Close'][:train_size], df['Close'][train_size:]
    
#     # Detailed NaN check for input data
#     print(f"Train size: {len(train)}, NaN count: {train.isna().sum()}")
#     print(f"Test size: {len(test)}, NaN count: {test.isna().sum()}")
#     if train.isna().any():
#         print("Cleaning train data...")
#         train = train.dropna()
#     if test.isna().any():
#         print("Cleaning test data...")
#         test = test.dropna()
#     if len(train) == 0 or len(test) == 0:
#         raise ValueError("Train or test data is empty after NaN removal.")
    
#     # Train model
#     model = auto_arima(train, seasonal=False, stepwise=True, trace=True)
#     fitted_model = model.fit(train)
#     predictions = fitted_model.predict(n_periods=len(test))
    
#     # Check predictions
#     print(f"Predictions length: {len(predictions)}, NaN count: {np.isnan(predictions).sum()}")
#     if np.isnan(predictions).any():
#         print("Warning: Predictions contain NaN. Replacing with last train value...")
#         predictions = np.nan_to_num(predictions, nan=train.iloc[-1])
    
#     # Align predictions with test index safely
#     if len(predictions) != len(test):
#         raise ValueError(f"Length mismatch: Predictions ({len(predictions)}) vs Test ({len(test)})")
#     # Create DataFrame to preserve values during alignment
#     predictions_df = pd.DataFrame({'Predictions': predictions}, index=test.index)
#     predictions = predictions_df['Predictions']
    
#     # Final validation
#     print(f"Test NaN count after alignment: {test.isna().sum()}")
#     print(f"Predictions NaN count after alignment: {predictions.isna().sum()}")
#     if test.isna().any() or predictions.isna().any():
#         print("Error: NaN values persist after alignment. Data preview:")
#         print("Test head:", test.head())
#         print("Predictions head:", predictions.head())
#         raise ValueError("NaN values persist after alignment.")
    
#     # Calculate metrics
#     mae = mean_absolute_error(test, predictions)
#     rmse = np.sqrt(mean_squared_error(test, predictions))
#     mape = np.mean(np.abs((test - predictions) / test)) * 100
    
#     # Plot for PowerPoint
#     plt.figure(figsize=(10, 6))
#     plt.plot(test.index, test, label='Actual')
#     plt.plot(test.index, predictions, label='Predicted', color='orange')
#     plt.title('TSLA ARIMA Predictions vs Actual')
#     plt.xlabel('Date')
#     plt.ylabel('Price (USD)')
#     plt.legend()
#     os.makedirs('reports', exist_ok=True)
#     plt.savefig('reports/TSLA_predictions.png')
#     plt.close()
    
#     return fitted_model, predictions, test, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# def train_sarima(df, train_split=0.8):
#     """Train a SARIMA model."""
#     train_size = int(len(df) * train_split)
#     train, test = df['Close'][:train_size], df['Close'][train_size:]
    
#     model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 252))
#     fitted_model = model.fit(disp=False)
#     predictions = fitted_model.forecast(steps=len(test))
    
#     mae = mean_absolute_error(test, predictions)
#     rmse = np.sqrt(mean_squared_error(test, predictions))
#     mape = np.mean(np.abs((test - predictions) / test)) * 100
    
#     return fitted_model, predictions, test, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# if __name__ == "__main__":
#     df = pd.read_csv("data/processed/TSLA_processed.csv", index_col='Date', parse_dates=True)
#     print("Initial Data Check:")
#     print(f"Close NaN count: {df['Close'].isna().sum()}")
#     if df['Close'].isna().any():
#         print("Cleaning initial data...")
#         df['Close'] = df['Close'].ffill().bfill()
    
#     model, predictions, test, metrics = train_arima(df)
#     print("ARIMA Metrics:", metrics)