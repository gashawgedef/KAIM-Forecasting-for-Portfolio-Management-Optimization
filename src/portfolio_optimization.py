import pandas as pd
import numpy as np
from scipy.optimize import minimize

def load_portfolio_data():
    """Load data for all assets."""
    tickers = ["TSLA", "BND", "SPY"]
    df = pd.DataFrame()
    for ticker in tickers:
        temp_df = pd.read_csv(f"../data/processed/{ticker}_processed.csv", index_col='Date')
        df[ticker] = temp_df['Close']
    return df

def portfolio_performance(weights, returns):
    """Calculate portfolio return and volatility."""
    port_return = np.sum(returns.mean() * weights) * 252  # Annualized
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return port_return, port_vol

def optimize_portfolio(df):
    """Optimize portfolio for maximum Sharpe Ratio."""
    returns = df.pct_change().dropna()
    n_assets = len(df.columns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def neg_sharpe(weights):
        port_return, port_vol = portfolio_performance(weights, returns)
        return -port_return / port_vol  # Negative Sharpe for minimization
    
    result = minimize(neg_sharpe, n_assets * [1./n_assets], method='SLSQP', 
                      bounds=bounds, constraints=constraints)
    return result.x

if __name__ == "__main__":
    df = load_portfolio_data()
    optimal_weights = optimize_portfolio(df)
    print("Optimal Weights:", dict(zip(df.columns, optimal_weights)))