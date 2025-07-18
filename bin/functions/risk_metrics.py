# risk_metrics.py
import numpy as np
import pandas as pd

def calculate_portfolio_var_es(returns_df: pd.DataFrame, weights: np.ndarray, alpha: float = 0.05):
    # 保证输入为 NumPy 数组
    returns_array = returns_df.values
    
    # 计算组合收益率
    portfolio_returns = returns_array.dot(weights)

    # 计算 VaR 和 ES
    var = np.quantile(portfolio_returns, alpha)
    es = portfolio_returns[portfolio_returns <= var].mean()

    return {"VaR": var, "ES": es}
