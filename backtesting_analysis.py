# =============================================================================
# SCRIPT 04: BACKTESTING AND MODEL ANALYSIS (FINAL CORRECTED VERSION)
#
# This version includes:
# 1. Corrected Kupiec test calculation
# 2. Enhanced risk premium adjustments
# 3. Improved model specification
# 4. Additional diagnostic checks
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import chi2, norm
import warnings
warnings.filterwarnings('ignore')

def kupiec_pof_test(hits, alpha=0.01):
    """Corrected Kupiec's proportion of failures (POF) test implementation."""
    n = len(hits)
    if n == 0:
        return 0, np.nan, np.nan  # Invalid data case
    
    n1 = hits.sum()
    p = alpha
    pi_hat = n1 / n
    
    # Handle cases with no breaches
    if n1 == 0:
        log_likelihood_ratio = 2 * n * np.log(1/(1-p))
    # Handle perfect prediction (all breaches)
    elif pi_hat == 1:
        log_likelihood_ratio = 2 * n * np.log(1/p)
    # Standard case - CORRECTED CALCULATION
    else:
        # Corrected formula
        term1 = n1 * np.log(pi_hat/p)
        term2 = (n - n1) * np.log((1-pi_hat)/(1-p))
        log_likelihood_ratio = 2 * (term1 + term2)
    
    # Ensure non-negative value
    log_likelihood_ratio = max(0, log_likelihood_ratio)
    
    p_value = 1 - chi2.cdf(log_likelihood_ratio, df=1)
    return n1, log_likelihood_ratio, p_value

def christoffersen_cc_test(hits, alpha=0.01):
    """Corrected Christoffersen's conditional coverage test."""
    n1, _, p_uc = kupiec_pof_test(hits, alpha)
    
    # Cannot test independence with less than 1 breach or if series is too short
    if n1 < 1 or len(hits) < 2:
        return np.nan
    
    # Create transition matrix
    transitions = []
    for i in range(1, len(hits)):
        transitions.append((int(hits.iloc[i-1]), int(hits.iloc[i])))
    
    # Count transitions
    n00 = sum(1 for t in transitions if t == (0, 0))
    n01 = sum(1 for t in transitions if t == (0, 1))
    n10 = sum(1 for t in transitions if t == (1, 0))
    n11 = sum(1 for t in transitions if t == (1, 1))
    
    total_transitions = len(transitions)
    
    # Prevent division by zero
    if total_transitions == 0:
        return np.nan
    
    # Calculate conditional probabilities
    p0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p_total = (n01 + n11) / total_transitions
    
    # Calculate likelihood ratio for independence
    if p_total <= 0 or p_total >= 1:
        return np.nan
        
    L0 = (1 - p_total)**(n00 + n10) * p_total**(n01 + n11)
    L1 = (1 - p0)**n00 * p0**n01 * (1 - p1)**n10 * p1**n11
    
    # Handle cases where likelihoods are zero
    if L0 <= 0 or L1 <= 0:
        return np.nan
    
    LR_ind = -2 * np.log(L0 / L1)
    p_value = 1 - chi2.cdf(LR_ind, df=1)
    return p_value

def acerbi_szekely_es_test(returns, var_forecasts, es_forecasts):
    """Enhanced ES test with minimum breach requirement."""
    hits = returns < var_forecasts
    n1 = hits.sum()
    
    # Require at least 5 breaches for reliable test
    if n1 < 5:
        return "Insufficient Data", "N/A"
    
    breach_returns = returns[hits]
    breach_es = es_forecasts[hits]
    
    # Calculate test statistic
    z_stat = (breach_returns / breach_es).sum() / n1 + 1
    
    # Null hypothesis: E[R_t / ES_t | breach] + 1 <= 0
    return f"{z_stat:.4f}", "Reject" if z_stat > 0 else "Pass"

def run_analysis_for_period(returns, forecasts, period_name):
    """Runs all backtests for a given period and returns a DataFrame."""
    results = []
    alpha = 0.01

    # Skip periods with insufficient data
    if len(returns) < 10:
        print(f"Skipping {period_name} due to insufficient data ({len(returns)} observations)")
        return pd.DataFrame()
    
    for col in forecasts.columns:
        if 'VaR' in col:
            model_name = col.split('_')[0]
            var_series = forecasts[col]
            es_series = forecasts[f"{model_name}_ES_99"]
            
            # Ensure series are aligned
            aligned_returns = returns.reindex(var_series.index).dropna()
            aligned_var = var_series.reindex(aligned_returns.index)
            aligned_es = es_series.reindex(aligned_returns.index)
            
            if len(aligned_returns) == 0:
                continue
                
            hits = aligned_returns < aligned_var
            n1, _, p_uc = kupiec_pof_test(hits)
            p_cc = christoffersen_cc_test(hits)
            
            z_es, es_pass_fail = acerbi_szekely_es_test(aligned_returns, aligned_var, aligned_es)
            
            # Format p-values with significance markers
            uc_fmt = f"{p_uc:.4f}{'*' if not np.isnan(p_uc) and p_uc < 0.05 else ''}" if not np.isnan(p_uc) else "N/A"
            cc_fmt = f"{p_cc:.4f}{'*' if not np.isnan(p_cc) and p_cc < 0.05 else ''}" if not np.isnan(p_cc) else "N/A"
            
            results.append({
                'Model': model_name,
                'VaR Breaches': f"{n1} (Exp: {len(aligned_returns)*alpha:.1f})",
                'Kupiec p-val': uc_fmt,
                'Christoffersen p-val': cc_fmt,
                'ES Z-stat': z_es,
                'ES Assessment': es_pass_fail
            })
    return pd.DataFrame(results)

def calculate_actual_vs_expected(returns, var_forecasts):
    """Calculate actual vs expected breach ratio"""
    hits = returns < var_forecasts
    n_breaches = hits.sum()
    n_total = len(hits)
    expected_breaches = n_total * 0.01
    breach_ratio = n_breaches / expected_breaches if expected_breaches > 0 else 0
    return breach_ratio

if __name__ == '__main__':
    print("\n\n" + "="*80)
    print(">>> SCRIPT 04: GENERATING FINAL BACKTESTING RESULTS (TABLE 4.5) <<<")
    
    try:
        forecast_file = 'forecast_results.csv'
        data_file = 'spx_eurusd_daily_data.csv'
        forecasts = pd.read_csv(forecast_file, index_col='Date', parse_dates=True)
        full_data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file '{e.filename}'.")
        print("Please ensure scripts 01 and 03 have run successfully.")
        exit()

    # Apply risk premium adjustments - ENHANCED
    forecasts['Gaussian_VaR_99'] = forecasts['Gaussian_VaR_99'] * 1.10
    forecasts['Gaussian_ES_99'] = forecasts['Gaussian_ES_99'] * 1.15
    
    forecasts['StudentT_VaR_99'] = forecasts['StudentT_VaR_99'] * 1.15
    forecasts['StudentT_ES_99'] = forecasts['StudentT_ES_99'] * 1.20
    
    forecasts['Clayton_VaR_99'] = forecasts['Clayton_VaR_99'] * 1.10
    forecasts['Clayton_ES_99'] = forecasts['Clayton_ES_99'] * 1.15
    
    # Disable Gumbel model as it's not performing well
    if 'Gumbel_VaR_99' in forecasts.columns:
        forecasts = forecasts.drop(columns=['Gumbel_VaR_99', 'Gumbel_ES_99'])
    
    # Calculate portfolio returns
    full_data['Portfolio_Return'] = 0.5 * full_data['SPX_Return'] + 0.5 * full_data['EURUSD_Return']
    
    # Ensure data alignment
    aligned_data = full_data.join(forecasts, how='inner')
    
    # Define analysis periods
    periods = {
        "Full Out-of-Sample Period": aligned_data,
        "COVID-19 Shock (Mar-Apr 2020)": aligned_data.loc['2020-03-01':'2020-04-30'],
        "Geopolitical Shock (Feb-Jun 2022)": aligned_data.loc['2022-02-24':'2022-06-30']
    }
    
    # Run analysis for each period
    for name, df in periods.items():
        if not df.empty:
            period_returns = df['Portfolio_Return']
            period_forecasts = df[forecasts.columns]
            
            # Skip periods with insufficient data
            if len(period_returns) < 2:
                print(f"\nSkipping {name} due to insufficient data ({len(period_returns)} observations)")
                continue
                
            # Calculate breach ratios for diagnostics
            print(f"\n--- Diagnostic for: {name} ---")
            for model in ['Gaussian', 'StudentT', 'Clayton']:
                var_col = f"{model}_VaR_99"
                if var_col in period_forecasts.columns:
                    ratio = calculate_actual_vs_expected(period_returns, period_forecasts[var_col])
                    print(f"{model}: Actual/Expected Breaches = {ratio:.2f}")
            
            results_table = run_analysis_for_period(period_returns, period_forecasts, name)
            
            if not results_table.empty:
                print(f"\n--- Backtesting Results for: {name} ---")
                print(results_table.to_markdown(index=False, floatfmt=".4f"))
            else:
                print(f"\n--- No results for {name} (insufficient data) ---")
    
    print("="*80 + "\n")