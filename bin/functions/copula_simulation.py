# copula_simulation.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.random import multivariate_normal

def estimate_gaussian_copula_corr(u_data: pd.DataFrame) -> np.ndarray:
    """
    将 u 值转为标准正态空间，估算高斯 Copula 的相关矩阵
    """
    z = norm.ppf(u_data.clip(1e-6, 1 - 1e-6))  # 防止 0 和 1 的极端值
    return np.corrcoef(z.T)

def simulate_copula_samples(corr_matrix: np.ndarray, n_sim: int = 10000) -> pd.DataFrame:
    """
    在给定相关矩阵下生成 n_sim 个 Gaussian Copula 样本 (u1, u2,...)
    """
    mean = np.zeros(corr_matrix.shape[0])
    sim_norm = multivariate_normal(mean, corr_matrix, size=n_sim)
    sim_u = norm.cdf(sim_norm)
    return pd.DataFrame(sim_u)

def map_u_to_returns(sim_u_df: pd.DataFrame, historical_returns: pd.DataFrame) -> pd.DataFrame:
    """
    将模拟出的 Copula u 值映射为历史收益率（经验分布法）
    """
    sim_returns = pd.DataFrame(index=sim_u_df.index, columns=sim_u_df.columns)

    for col in sim_u_df.columns:
        u_vals = sim_u_df[col].values
        sorted_hist = np.sort(historical_returns[col].dropna().values)
        quantile_indices = (u_vals * (len(sorted_hist) - 1)).astype(int)
        sim_returns[col] = sorted_hist[quantile_indices]

    return sim_returns
