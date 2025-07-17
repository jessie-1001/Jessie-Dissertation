# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION AND DIAGNOSTICS (REVISED)
#
# This script now uses a more robust ARMA(1,1)-GARCH(1,1)-t model to
# better capture the dynamics of the return series.
# =============================================================================

import pandas as pd
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import t
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def fit_and_diagnose_garch(return_series, asset_name):
    """
    Fits an ARMA(1,1)-GARCH(1,1)-t model and performs diagnostics.
    """
    print(f"--- Fitting ARMA(1,1)-GARCH(1,1)-t model for {asset_name} ---")
    
    # Specify the ARMA(1,1)-GARCH(1,1) model with Student's t distribution
    # Note: mean='ARX', lags=1 handles the AR part. o=1 adds the MA part.
    model = arch_model(return_series, mean='ARX', lags=1, vol='Garch', p=1, o=1, q=1, dist='t')
    
    result = model.fit(update_freq=0, disp='off')
    print(result.summary())
    
    print(f"\n--- Diagnostic tests for {asset_name} residuals ---")
    std_resid = result.std_resid
    lj_resid = sm.stats.acorr_ljungbox(std_resid, lags=[10], return_df=True)
    lj_sq_resid = sm.stats.acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
    print("Ljung-Box test on standardized residuals:\n", lj_resid)
    print("\nLjung-Box test on squared standardized residuals:\n", lj_sq_resid)
    print("-" * 50)
    
    return result

if __name__ == '__main__':
    input_file = 'spx_eurusd_daily_data.csv'
    data = pd.read_csv(input_file, index_col='Date', parse_dates=True)
    
    in_sample_end = '2019-12-31'
    in_sample_data = data.loc[:in_sample_end]
    
    spx_returns = in_sample_data['SPX_Return'] * 100
    eurusd_returns = in_sample_data['EURUSD_Return'] * 100

    result_spx = fit_and_diagnose_garch(spx_returns, 'S&P 500')
    result_eurusd = fit_and_diagnose_garch(eurusd_returns, 'EUR/USD')
    
    joblib.dump(result_spx, 'fitted_garch_spx.joblib')
    joblib.dump(result_eurusd, 'fitted_garch_eurusd.joblib')
    print("Fitted GARCH models have been saved.")

    u_spx = t.cdf(result_spx.std_resid, df=result_spx.params['nu'])
    u_eurusd = t.cdf(result_eurusd.std_resid, df=result_eurusd.params['nu'])

    copula_data = pd.DataFrame({
        'u_spx': u_spx,
        'u_eurusd': u_eurusd
    }, index=spx_returns.index)

    copula_output_file = 'copula_input_data.csv'
    copula_data.to_csv(copula_output_file)

    print(f"\nData ready for copula modeling has been saved to '{copula_output_file}'.")
    print(copula_data.head())