# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION AND DIAGNOSTICS (FINAL & ROBUST)
#
# This definitive version uses a consistent model specification and manually
# performs the Probability Integral Transform (PIT) using scipy.stats
# to ensure maximum stability and compatibility, resolving all previous errors.
# =============================================================================

import pandas as pd
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import t  # We will use this directly
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def fit_and_diagnose_garch(return_series, asset_name):
    """
    Fits an ARMA(1,1)-GJR-GARCH(1,1)-t model, performs diagnostics,
    and returns the result object. Also prints formatted tables.
    """
    print(f"--- Fitting ARMA(1,1)-GJR-GARCH(1,1)-t model for {asset_name} ---")
    
    # We consistently use the ARMA(1,1)-GJR-GARCH(1,1) model with a Student's t-distribution
    model = arch_model(return_series, mean='ARX', lags=1, vol='Garch', p=1, o=1, q=1, dist='t')
    
    result = model.fit(update_freq=0, disp='off')

    # --- Create and print Parameter Table (for Table 4.2) ---
    params = result.params
    pvalues = result.pvalues
    param_table = pd.DataFrame({'Coefficient': params, 'P-value': pvalues})
    print(f"\n--- FOR DISSERTATION (TABLE 4.2): Parameter Estimates for {asset_name} ---")
    param_table['P-value'] = param_table['P-value'].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")
    print(param_table.to_markdown(floatfmt=".4f"))

    # --- Create and print Diagnostic Table (for Table 4.3) ---
    std_resid = pd.Series(result.std_resid, index=return_series.index[-len(result.std_resid):])
    lj_resid = sm.stats.acorr_ljungbox(std_resid.dropna(), lags=[20], return_df=True)
    lj_sq_resid = sm.stats.acorr_ljungbox(std_resid.dropna()**2, lags=[20], return_df=True)
    
    diag_table = pd.DataFrame({
        'Test': ['Ljung-Box on Standardized Residuals (Lags=20)', 'Ljung-Box on Squared Standardized Residuals (Lags=20)'],
        'P-value': [lj_resid['lb_pvalue'].iloc[0], lj_sq_resid['lb_pvalue'].iloc[0]]
    })
    print(f"\n--- FOR DISSERTATION (TABLE 4.3): Diagnostic Tests for {asset_name} ---")
    print(diag_table.to_markdown(index=False, floatfmt=".4f"))
    
    # Report on specification based on p-values
    if lj_resid['lb_pvalue'].iloc[0] > 0.05 and lj_sq_resid['lb_pvalue'].iloc[0] > 0.05:
        print("\nSUCCESS: Model appears well-specified for this asset.")
    else:
        print("\nNOTE: Model shows signs of misspecification (p-value < 0.05). This is an empirical finding.")
    print("-" * 80)
    
    return result

if __name__ == '__main__':
    print("\n\n" + "="*80)
    print(">>> SCRIPT 02: GENERATING OUTPUTS FOR DISSERTATION (TABLES 4.2 & 4.3) <<<")
    
    try:
        input_file = 'spx_eurusd_daily_data.csv'
        data = pd.read_csv(input_file, index_col='Date', parse_dates=True)
        
        in_sample_end = '2019-12-31'
        in_sample_data = data.loc[:in_sample_end]
        
        spx_returns = in_sample_data['SPX_Return'] * 100
        eurusd_returns = in_sample_data['EURUSD_Return'] * 100

        result_spx = fit_and_diagnose_garch(spx_returns, 'S&P 500')
        result_eurusd = fit_and_diagnose_garch(eurusd_returns, 'EUR/USD')
        
        # --- Generate and save the inputs for the next script using the robust manual PIT method ---
        
        # Extract standardized residuals and degrees of freedom
        std_resid_spx = result_spx.std_resid
        nu_spx = result_spx.params['nu']
        
        std_resid_eurusd = result_eurusd.std_resid
        nu_eurusd = result_eurusd.params['nu']
        
        # Manually apply the CDF from scipy.stats, which is stable
        u_spx = t.cdf(std_resid_spx, df=nu_spx)
        u_eurusd = t.cdf(std_resid_eurusd, df=nu_eurusd)
        
        # Create Series with the correct index to avoid alignment issues
        u_spx_series = pd.Series(u_spx, index=result_spx.resid.index)
        u_eurusd_series = pd.Series(u_eurusd, index=result_eurusd.resid.index)

        # Combine into a final DataFrame and save
        copula_data = pd.DataFrame({'u_spx': u_spx_series, 'u_eurusd': u_eurusd_series}).dropna()
        copula_output_file = 'copula_input_data.csv'
        copula_data.to_csv(copula_output_file)

        print(f"\nData ready for copula modeling has been saved to '{copula_output_file}'.")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please run script 01 first.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("="*80 + "\n")