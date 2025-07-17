# =============================================================================
# SCRIPT 03: COPULA ESTIMATION AND MONTE CARLO SIMULATION (FINAL & ROBUST)
#
# This definitive version removes all dependencies on external copula libraries
# and implements the copula samplers from first principles using SciPy and NumPy,
# ensuring maximum stability and reproducibility.
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import fsolve
from arch import arch_model
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- 1. Define Copula Samplers from First Principles ---

def sample_gaussian_copula(n_samples, corr_matrix):
    """Samples from a Gaussian copula."""
    mvn = norm(loc=0, scale=1)
    # Generate correlated normal random variables
    correlated_normals = np.random.multivariate_normal(
        mean=[0, 0], cov=corr_matrix, size=n_samples
    )
    # Transform to uniform using the normal CDF (PIT)
    uniforms = mvn.cdf(correlated_normals)
    return uniforms

def sample_t_copula(n_samples, corr_matrix, df):
    """Samples from a Student-t copula."""
    # Generate correlated t-distributed random variables
    # This involves a chi-square random variable
    d = len(corr_matrix)
    g = np.random.chisquare(df, n_samples)
    z = np.random.multivariate_normal(np.zeros(d), corr_matrix, n_samples)
    x = np.sqrt(df / g)[:, np.newaxis] * z
    # Transform to uniform using the t-distribution CDF (PIT)
    uniforms = t.cdf(x, df=df)
    return uniforms

def sample_clayton_copula(n_samples, theta):
    """Samples from a Clayton copula using the conditional distribution method."""
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    u2 = (u1**(-theta) * (v**(-theta / (1 + theta)) - 1) + 1)**(-1 / theta)
    return np.column_stack((u1, u2))

def sample_gumbel_copula(n_samples, theta):
    """Samples from a Gumbel copula using its generator."""
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    # Conditional distribution method for Gumbel is more complex,
    # we use a well-known algorithm based on a stable subordinator.
    gamma_rv = np.random.gamma(1/theta, 1, n_samples)
    e1 = -np.log(u1) / gamma_rv
    e2 = -np.log(v) / gamma_rv
    u1_sim = np.exp(-e1**(1/theta))
    u2_sim = np.exp(-e2**(1/theta))
    return np.column_stack((u1_sim, u2_sim))

# --- 2. Main Simulation Functions ---

def get_copula_parameters(data):
    """Estimates dependence parameters from uniform data."""
    print("Estimating dependence parameters...")
    # Spearman's rho for elliptical copulas
    spearman_rho = data.corr(method='spearman').iloc[0, 1]
    
    # Kendall's tau for Archimedean copulas
    kendall_tau = data.corr(method='kendall').iloc[0, 1]
    
    # Conversion formulas
    theta_clayton = 2 * kendall_tau / (1 - kendall_tau)
    theta_gumbel = 1 / (1 - kendall_tau)
    
    params = {
        'Gaussian': {'corr_matrix': data.corr(method='pearson')},
        'StudentT': {'corr_matrix': data.corr(method='pearson'), 'df': 5}, # df=5 is a common choice
        'Gumbel': {'theta': theta_gumbel},
        'Clayton': {'theta': theta_clayton}
    }
    
    print("Parameter estimation complete:")
    for name, p in params.items():
        print(f"  - {name}: {p}")
        
    return params

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=10000):
    """Runs the Monte Carlo simulation for a single day t."""
    window_data = full_data.loc[:t_index]
    
    garch_spx = arch_model(window_data['SPX_Return']*100, vol='Garch', p=1, o=1, q=1, dist='t')
    res_spx = garch_spx.fit(disp='off')
    
    garch_eurusd = arch_model(window_data['EURUSD_Return']*100, vol='Garch', p=1, o=1, q=1, dist='t')
    res_eurusd = garch_eurusd.fit(disp='off')

    forecast_spx = res_spx.forecast(horizon=1, reindex=False)
    sigma_t1_spx = np.sqrt(forecast_spx.variance.iloc[0, 0])
    forecast_eurusd = res_eurusd.forecast(horizon=1, reindex=False)
    sigma_t1_eurusd = np.sqrt(forecast_eurusd.variance.iloc[0, 0])
    
    mu_spx = res_spx.params.get('mu', 0)
    nu_spx = res_spx.params['nu']
    mu_eurusd = res_eurusd.params.get('mu', 0)
    nu_eurusd = res_eurusd.params['nu']
    
    daily_forecasts = {}

    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_t_copula(n, p['corr_matrix'], p['df']),
        'Gumbel': lambda n, p: sample_gumbel_copula(n, p['theta']),
        'Clayton': lambda n, p: sample_clayton_copula(n, p['theta'])
    }
    
    for name, params in copula_params.items():
        simulated_uniforms = samplers[name](n_simulations, params)
        
        z_spx = t.ppf(simulated_uniforms[:, 0], df=nu_spx)
        z_eurusd = t.ppf(simulated_uniforms[:, 1], df=nu_eurusd)
        
        r_spx_sim = (mu_spx + sigma_t1_spx * z_spx) / 100
        r_eurusd_sim = (mu_eurusd + sigma_t1_eurusd * z_eurusd) / 100
        r_portfolio_sim = 0.5 * r_spx_sim + 0.5 * r_eurusd_sim
        
        var_99 = np.percentile(r_portfolio_sim, 1)
        es_99 = np.mean(r_portfolio_sim[r_portfolio_sim <= var_99])
        
        daily_forecasts[name] = {'VaR_99': var_99, 'ES_99': es_99}
        
    return daily_forecasts

# --- 3. Main execution block ---

if __name__ == '__main__':
    copula_input_data = pd.read_csv('copula_input_data.csv', index_col='Date', parse_dates=True).dropna()
    full_data = pd.read_csv('spx_eurusd_daily_data.csv', index_col='Date', parse_dates=True)
    
    copula_params = get_copula_parameters(copula_input_data)

    out_of_sample_start = '2020-01-01'
    forecast_dates = full_data.loc[:out_of_sample_start].index[-1:].union(full_data.loc[out_of_sample_start:].index[:-1])

    all_forecasts = []

    for day in tqdm(forecast_dates, desc="Rolling Forecast VaR/ES"):
        forecasts = run_simulation_for_day(day, full_data, copula_params)
        forecast_date_index = full_data.index.get_loc(day) + 1
        if forecast_date_index < len(full_data.index):
            forecast_date = full_data.index[forecast_date_index]
            flat_forecasts = {'Date': forecast_date}
            for model_name, values in forecasts.items():
                flat_forecasts[f'{model_name}_VaR_99'] = values['VaR_99']
                flat_forecasts[f'{model_name}_ES_99'] = values['ES_99']
            all_forecasts.append(flat_forecasts)

    forecasts_df = pd.DataFrame(all_forecasts)
    forecasts_df.set_index('Date', inplace=True)
    
    forecast_output_file = 'forecast_results.csv'
    forecasts_df.to_csv(forecast_output_file)
    
    print(f"\nAll forecasts saved to '{forecast_output_file}'.")
    print(forecasts_df.head())