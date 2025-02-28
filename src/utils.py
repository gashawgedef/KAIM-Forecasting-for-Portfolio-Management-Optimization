import matplotlib.pyplot as plt
import numpy as np

def calculate_var(returns, confidence_level=0.95):
    """Calculate Value at Risk (VaR)."""
    return np.percentile(returns.dropna(), (1 - confidence_level) * 100)

def save_plot(fig, filename, output_dir="../reports/final_submission/"):
    """Save matplotlib figure to file."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)