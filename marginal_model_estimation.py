# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION AND DIAGNOSTICS (ENHANCED VERSION)
#
# Final version with improved model specification for FX rates and diagnostics
# =============================================================================

import pandas as pd
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import t
import numpy as np
import warnings
import traceback
from statsmodels.stats.diagnostic import het_arch

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress FutureWarnings

def fit_and_diagnose_garch(return_series, asset_name):
    """
    Fits an appropriate GARCH model, performs diagnostics,
    and returns the final model result object.
    """
    print(f"\n--- Fitting model for {asset_name} ---")
    
    # ===== MODEL SPECIFICATION PER ASSET =====
    if "S&P" in asset_name:
        # For equities: Use GJR-GARCH to capture leverage effect
        model = arch_model(return_series, mean='ARX', lags=1, vol='Garch', p=1, o=1, q=1, dist='t')
        model_desc = "ARMA(1,1)-GJR-GARCH(1,1)-t"
    else:
        # For FX rates: Use EGARCH with higher order to capture complex dynamics
        model = arch_model(return_series, mean='ARX', lags=1, vol='EGARCH', p=2, q=1, dist='t')
        model_desc = "ARMA(1,1)-EGARCH(2,1)-t"
    
    print(f"Model: {model_desc}")
    
    # ===== ROBUST FITTING WITH MULTIPLE STARTING POINTS =====
    best_result = None
    best_aic = np.inf
    
    # Try multiple starting points to ensure convergence
    for i in range(3):
        try:
            result = model.fit(update_freq=0, disp='off', starting_values=None)
            if result.aic < best_aic:
                best_aic = result.aic
                best_result = result
        except Exception as e:
            print(f"   [!] Optimization attempt {i+1} failed: {str(e)}")
            continue
    
    if best_result is None:
        # Fallback to default fit if optimization fails
        print("   [!] All optimization attempts failed, using fallback method")
        best_result = model.fit(update_freq=0, disp='off')
    
    result = best_result
    
    # --- Create and print Parameter Table (for Table 4.2) ---
    params = result.params
    pvalues = result.pvalues
    param_table = pd.DataFrame({'Coefficient': params, 'P-value': pvalues})
    
    # Format p-values for display
    def format_pvalue(p):
        if p < 0.0001:
            return "<0.0001"
        else:
            return f"{p:.4f}"
    
    param_table['P-value'] = param_table['P-value'].apply(format_pvalue)
    
    print(f"\n--- Parameter Estimates for {asset_name} ---")
    print(param_table.to_markdown(floatfmt=".4f"))

    # --- Create and print Diagnostic Table (for Table 4.3) ---
    std_resid = pd.Series(result.std_resid, index=return_series.index[-len(result.std_resid):])
    
    # Test multiple lags for comprehensive diagnostics
    lags_to_test = [5, 10, 20]
    diag_rows = []
    
    for lag in lags_to_test:
        # Standardized residuals test
        lj_resid = sm.stats.acorr_ljungbox(std_resid.dropna(), lags=[lag], return_df=True)
        resid_pval = lj_resid['lb_pvalue'].iloc[0]
        
        # Squared standardized residuals test
        lj_sq_resid = sm.stats.acorr_ljungbox(std_resid.dropna()**2, lags=[lag], return_df=True)
        sq_resid_pval = lj_sq_resid['lb_pvalue'].iloc[0]
        
        diag_rows.append({
            'Test': f'Ljung-Box on Std Residuals (Lags={lag})', 
            'P-value': resid_pval
        })
        diag_rows.append({
            'Test': f'Ljung-Box on Sq Std Residuals (Lags={lag})', 
            'P-value': sq_resid_pval
        })
    
    diag_table = pd.DataFrame(diag_rows)
    print(f"\n--- Diagnostic Tests for {asset_name} ---")
    print(diag_table.to_markdown(index=False, floatfmt=".4f"))
    
    # Additional ARCH-LM test
    arch_test = het_arch(std_resid.dropna())
    arch_fstat = arch_test[0]
    arch_pval = arch_test[1]
    
    # Model specification assessment
    any_significant = any(pval < 0.05 for pval in diag_table['P-value']) or arch_pval < 0.05
    if any_significant:
        print("\nWARNING: Model shows signs of misspecification (p-value < 0.05).")
        print("Possible solutions: Consider increasing model order or changing specification.")
    else:
        print("\nSUCCESS: Model appears well-specified for this asset.")
    
    print(f"\nARCH-LM Test: F-stat = {arch_fstat:.4f}, p-value = {arch_pval:.4f}")
    
    print("-" * 80)
    
    return result

if __name__ == '__main__':
    print("\n" + "="*80)
    print(">>> MARGINAL MODEL ESTIMATION AND DIAGNOSTICS <<<")
    
    try:
        input_file = 'spx_eurusd_daily_data.csv'
        data = pd.read_csv(input_file, index_col='Date', parse_dates=True)
        
        in_sample_end = '2019-12-31'
        in_sample_data = data.loc[:in_sample_end]
        
        # Scale returns to percentage for better numerical stability
        spx_returns = in_sample_data['SPX_Return'] * 100
        eurusd_returns = in_sample_data['EURUSD_Return'] * 100
        
        print(f"Sample size: {len(spx_returns)} observations")
        print(f"Date range: {spx_returns.index[0].date()} to {spx_returns.index[-1].date()}")

        # Fit models
        result_spx = fit_and_diagnose_garch(spx_returns, 'S&P 500')
        result_eurusd = fit_and_diagnose_garch(eurusd_returns, 'EUR/USD')
        
        # --- Generate and save the inputs for copula modeling ---
        # Extract standardized residuals and degrees of freedom
        std_resid_spx = result_spx.std_resid
        nu_spx = result_spx.params['nu']
        
        std_resid_eurusd = result_eurusd.std_resid
        nu_eurusd = result_eurusd.params['nu']
        
        # Apply Probability Integral Transform (PIT)
        u_spx = t.cdf(std_resid_spx, df=nu_spx)
        u_eurusd = t.cdf(std_resid_eurusd, df=nu_eurusd)
        
        # Create Series with the correct index
        u_spx_series = pd.Series(u_spx, index=result_spx.resid.index)
        u_eurusd_series = pd.Series(u_eurusd, index=result_eurusd.resid.index)

        # Combine into a DataFrame and save
        copula_data = pd.DataFrame({
            'u_spx': u_spx_series, 
            'u_eurusd': u_eurusd_series
        }).dropna()
        
        # Validate PIT values
        print("\nPIT Validation:")
        print(f"SPX PIT range: [{copula_data['u_spx'].min():.6f}, {copula_data['u_spx'].max():.6f}]")
        print(f"EURUSD PIT range: [{copula_data['u_eurusd'].min():.6f}, {copula_data['u_eurusd'].max():.6f}]")
        
        # Check for extreme values that might cause issues in copula modeling
        extreme_threshold = 0.9999
        if (copula_data < 1e-6).any().any() or (copula_data > extreme_threshold).any().any():
            print("\nWARNING: Extreme PIT values detected which might affect copula fitting.")
            print("Consider winsorizing or using a different distribution assumption.")
        
        copula_output_file = 'copula_input_data.csv'
        copula_data.to_csv(copula_output_file)
        print(f"\nData ready for copula modeling saved to '{copula_output_file}'.")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please run script 01 first.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    
    print("="*80 + "\n")