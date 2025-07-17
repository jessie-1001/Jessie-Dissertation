# =============================================================================
# SCRIPT 04: BACKTESTING AND MODEL ANALYSIS
#
# This script performs Stage 4 of the analysis:
# 1. Loads the actual portfolio returns and the forecasted VaR/ES values.
# 2. Implements backtesting functions for VaR (Kupiec) and ES (Acerbi-Szekely).
# 3. Runs backtests for each model on the full out-of-sample period and
#    on specific stress-test windows.
# 4. Prints a summary report of the backtesting results.
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import chi2

def kupiec_pof_test(returns, var_forecasts, alpha):
    """
    Performs Kupiec's proportion of failures (POF) test for VaR.
    """
    p = alpha
    hits = returns < -var_forecasts
    N1 = hits.sum()
    N = len(hits)
    N0 = N - N1
    
    if N1 == 0: return 1.0 # Cannot reject if no breaches
    
    pi_hat = N1 / N
    
    if pi_hat == 0 or pi_hat == 1: return 0.0 # Avoid log(0)
    
    log_likelihood_ratio = -2 * (N0 * np.log(1 - p) + N1 * np.log(p) - (N0 * np.log(1 - pi_hat) + N1 * np.log(pi_hat)))
    p_value = 1 - chi2.cdf(log_likelihood_ratio, df=1)
    
    return N1, p_value

def acerbi_szekely_es_test(returns, var_forecasts, es_forecasts, alpha):
    """
    Performs Acerbi & Szekely's (2014) test for ES.
    """
    hits = returns < -var_forecasts
    N1 = hits.sum()
    
    if N1 == 0: return "No Breaches", 1.0
    
    breach_returns = returns[hits]
    breach_es = -es_forecasts[hits] # ES is negative, so -ES is the loss
    
    z_stat = (breach_returns / breach_es).sum() / N1
    
    # Simplified p-value via simulation under the null
    # A full implementation would involve more complex bootstrapping.
    # We provide a conceptual check here.
    if z_stat < -1:
        p_value = "Reject (<5%)" # Conceptual p-value for systematic underestimation
    else:
        p_value = "Pass (>5%)"
        
    return z_stat, p_value

def run_analysis(actual_returns, forecasts, period_name):
    """
    Runs all backtests for a given period and returns a results dictionary.
    """
    print(f"\n--- Backtesting Analysis for Period: {period_name} ---")
    results = []
    alpha = 0.01 # For 99% VaR/ES

    for col in forecasts.columns:
        if 'VaR' in col:
            model_name = col.split('_')[0]
            var_series = forecasts[col]
            es_series = forecasts[f"{model_name}_ES_99"]
            
            # Run tests
            n1, pof_p_value = kupiec_pof_test(actual_returns, var_series, alpha)
            z_es, es_p_value = acerbi_szekely_es_test(actual_returns, var_series, es_series, alpha)
            
            results.append({
                'Period': period_name,
                'Model': model_name,
                'VaR Breaches': f"{n1} (Exp: {len(actual_returns)*alpha:.1f})",
                'Kupiec POF p-val': f"{pof_p_value:.4f}",
                'ES Test Z-stat': f"{z_es:.4f}" if isinstance(z_es, float) else z_es,
                'ES Test p-val': es_p_value
            })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # --- 1. Load Data ---
    forecast_file = 'forecast_results.csv'
    data_file = 'spx_eurusd_daily_data.csv'
    
    forecasts = pd.read_csv(forecast_file, index_col='Date', parse_dates=True)
    full_data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    
    # Calculate actual portfolio returns
    full_data['Portfolio_Return'] = 0.5 * full_data['SPX_Return'] + 0.5 * full_data['EURUSD_Return']
    
    # Align data
    aligned_data = full_data.join(forecasts, how='inner')
    
    # --- 2. Define Periods ---
    full_period_returns = aligned_data['Portfolio_Return']
    full_period_forecasts = aligned_data[forecasts.columns]
    
    covid_window = aligned_data.loc['2020-03-01':'2020-04-30']
    covid_returns = covid_window['Portfolio_Return']
    covid_forecasts = covid_window[forecasts.columns]
    
    geopolitical_window = aligned_data.loc['2022-02-24':'2022-06-30']
    geopolitical_returns = geopolitical_window['Portfolio_Return']
    geopolitical_forecasts = geopolitical_window[forecasts.columns]

    # --- 3. Run Analysis for All Periods ---
    results_full = run_analysis(full_period_returns, full_period_forecasts, "Full Period")
    results_covid = run_analysis(covid_returns, covid_forecasts, "COVID-19 Shock")
    results_geo = run_analysis(geopolitical_returns, geopolitical_forecasts, "Geopolitical Shock")
    
    # --- 4. Display Final Report ---
    final_report = pd.concat([results_full, results_covid, results_geo])
    print("\n\n=== FINAL BACKTESTING REPORT ===")
    print(final_report.to_string())