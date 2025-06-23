import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def fit_garch_models(log_returns: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    garch_results = {}
    residuals = pd.DataFrame(index=log_returns.index)
    volatility = pd.DataFrame(index=log_returns.index)

    for col in log_returns.columns:
        print(f"--- Fitting GARCH(1,1) for {col} ---")
        model = arch_model(log_returns[col], vol='Garch', p=1, q=1, dist='normal')
        result = model.fit(disp='off')
        garch_results[col] = result
        residuals[col] = result.resid
        volatility[col] = result.conditional_volatility
        print(result.summary())
        print("\n")

    return garch_results, residuals, volatility


def plot_residual_analysis(residuals: pd.DataFrame) -> None:
    for col in residuals.columns:
        resid = residuals[col].dropna()

        plt.figure(figsize=(10, 4))
        sns.histplot(resid, kde=True, stat="density", bins=60, label="Residuals")
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), label="Normal PDF", color="red")
        plt.title(f"Histogram & KDE of Residuals: {col}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        stats.probplot(resid, dist="norm", plot=plt)
        plt.title(f"QQ Plot of Residuals: {col}")
        plt.tight_layout()
        plt.show()


def generate_copula_inputs(residuals: pd.DataFrame) -> pd.DataFrame:
    for col in residuals.columns:
        resid = residuals[col].dropna()
        df_t, loc_t, scale_t = stats.t.fit(resid)
        u_empirical = stats.rankdata(resid) / (len(resid) + 1)
        u_t_dist = stats.t.cdf(resid, df=df_t, loc=loc_t, scale=scale_t)
        residuals[f"{col}_u_empirical"] = u_empirical
        residuals[f"{col}_u_t"] = u_t_dist

    cols_u_t = [col for col in residuals.columns if col.endswith("_u_t")]
    copula_data = residuals[cols_u_t].dropna().copy()
    copula_data.columns = [col.split("_u_t")[0] for col in cols_u_t]
    return copula_data
