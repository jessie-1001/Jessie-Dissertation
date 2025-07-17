# =============================================================================
# SCRIPT 03: COPULA ESTIMATION AND MONTE CARLO SIMULATION (DEFINITIVE VERSION)
#
# This definitive version incorporates all expert suggestions and fixes the final
# bug by manually performing the inverse transform (PPF), ensuring it runs
# successfully regardless of library version issues.
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, t
from arch import arch_model
from tqdm import tqdm
import warnings
from scipy.optimize import minimize
import pickle

warnings.filterwarnings('ignore')

# --- 1. Copula Samplers and Parameter Estimation ---

def sample_gaussian_copula(n_samples, corr_matrix):
    return norm.cdf(np.random.multivariate_normal(mean=[0, 0], cov=corr_matrix, size=n_samples))

def sample_t_copula(n_samples, corr_matrix, df):
    d = len(corr_matrix)
    g = np.random.chisquare(df, n_samples)
    z = np.random.multivariate_normal(np.zeros(d), corr_matrix, n_samples)
    x = np.sqrt(df / g)[:, np.newaxis] * z
    return t.cdf(x, df=df)

def sample_clayton_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    u2 = (u1**(-theta) * (v**(-theta / (1 + theta)) - 1) + 1)**(-1 / theta)
    return np.column_stack((u1, u2))

def sample_gumbel_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    gamma_rv = np.random.gamma(1/theta, 1, n_samples)
    e1 = -np.log(u1) / gamma_rv
    e2 = -np.log(v) / gamma_rv
    u1_sim = np.exp(-e1**(1/theta))
    u2_sim = np.exp(-e2**(1/theta))
    return np.column_stack((u1_sim, u2_sim))

def fit_t_copula_mle(data):
    """Estimates t-copula parameters (rho, df) using MLE."""
    def log_likelihood(params, data):
        rho, df = params[0], params[1]
        if not -0.999 < rho < 0.999 or df <= 2: return np.inf
        try:
            t_dist = t(df)
            x1 = t_dist.ppf(data.iloc[:, 0].clip(1e-6, 1-1e-6))
            x2 = t_dist.ppf(data.iloc[:, 1].clip(1e-6, 1-1e-6))
            cov_matrix = np.array([[1, rho], [rho, 1]])
            # Manually calculate bivariate t-distribution log pdf
            const_term = -np.log(2 * np.pi) - 0.5 * np.log(1 - rho**2)
            main_term = -((df + 2) / 2) * np.log(1 + (x1**2 - 2 * rho * x1 * x2 + x2**2) / (df * (1 - rho**2)))
            log_pdf_copula = const_term + main_term
            log_pdf_marginals = t_dist.logpdf(x1) + t_dist.logpdf(x2)
            log_likelihood_val = np.sum(log_pdf_copula - log_pdf_marginals)
            return -log_likelihood_val
        except (ValueError, np.linalg.LinAlgError):
            return np.inf

    init_rho = data.corr(method='spearman').iloc[0,1]
    init_df = 6.0
    result = minimize(log_likelihood, [init_rho, init_df], args=(data,), method='L-BFGS-B', bounds=[(-0.99, 0.99), (2.1, 50)])

    if result.success:
        rho_est, df_est = result.x
        return {'corr_matrix': np.array([[1, rho_est], [rho_est, 1]]), 'df': df_est}
    else:
        print("T-copula MLE failed, using fallback parameters.")
        return {'corr_matrix': data.corr(method='pearson').values, 'df': 5}

def get_copula_parameters(data):
    """Estimates dependence parameters and prints the formatted table."""
    print("Estimating dependence parameters for the in-sample period...")
    kendall_tau = data.corr(method='kendall').iloc[0, 1]
    pearson_corr = data.corr(method='pearson').values
    
    theta_clayton = 2 * kendall_tau / (1 - kendall_tau) if (1 - kendall_tau) != 0 else np.inf
    theta_gumbel = 1 / (1 - kendall_tau) if (1 - kendall_tau) != 0 else np.inf
    if theta_clayton <= 0: theta_clayton = 0.001
    if theta_gumbel <= 1: theta_gumbel = 1.001

    t_copula_params = fit_t_copula_mle(data)
    
    params = {
        'Gaussian': {'corr_matrix': pearson_corr},
        'StudentT': t_copula_params,
        'Gumbel': {'theta': theta_gumbel},
        'Clayton': {'theta': theta_clayton}
    }
    
    param_data = {
        'Gaussian': {'ρ (Pearson)': pearson_corr[0,1]},
        'StudentT': {'ρ (MLE)': t_copula_params['corr_matrix'][0,1], 'ν (DoF, MLE)': t_copula_params['df']},
        'Gumbel': {'θ (from Kendall)': theta_gumbel},
        'Clayton': {'θ (from Kendall)': theta_clayton}
    }
    param_df = pd.DataFrame(param_data).T.fillna('')
    
    print("\n\n" + "="*80)
    print(">>> OUTPUT FOR DISSERTATION: TABLE 4.4 <<<")
    print(param_df.to_markdown(floatfmt=".4f"))
    print("="*80 + "\n")
        
    return params

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=10000):
    """Runs the Monte Carlo simulation for a single day t."""
    window_data = full_data.loc[:t_index]
    
    garch_spx = arch_model(window_data['SPX_Return']*100, mean='ARX', lags=1, vol='Garch', p=1, o=1, q=1, dist='t')
    res_spx = garch_spx.fit(disp='off')
    
    garch_eurusd = arch_model(window_data['EURUSD_Return']*100, mean='ARX', lags=1, vol='EGARCH', p=2, q=1, dist='t')
    res_eurusd = garch_eurusd.fit(disp='off')

    sigma_t1_spx = np.sqrt(res_spx.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    sigma_t1_eurusd = np.sqrt(res_eurusd.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    
    last_return_spx = window_data['SPX_Return'].iloc[-1] * 100
    last_return_eurusd = window_data['EURUSD_Return'].iloc[-1] * 100
    
    mu_spx_next = res_spx.params['Const'] + res_spx.params['SPX_Return[1]'] * last_return_spx
    mu_eurusd_next = res_eurusd.params['Const'] + res_eurusd.params[1] * last_return_eurusd
    
    daily_forecasts = {}
    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_t_copula(n, p['corr_matrix'], p['df']),
        'Gumbel': lambda n, p: sample_gumbel_copula(n, p['theta']),
        'Clayton': lambda n, p: sample_clayton_copula(n, p['theta'])
    }
    
    # Get degrees of freedom from the re-estimated models
    nu_spx = res_spx.params['nu']
    nu_eurusd = res_eurusd.params['nu']
    
    for name, params in copula_params.items():
        simulated_uniforms = samplers[name](n_simulations, params)
        
        # <<< START OF FINAL FIX >>>
        # Manually perform the inverse transform using scipy.stats.t.ppf
        # This is robust and bypasses any library versioning issues.
        z_spx = t.ppf(simulated_uniforms[:, 0], df=nu_spx)
        z_eurusd = t.ppf(simulated_uniforms[:, 1], df=nu_eurusd)
        # <<< END OF FINAL FIX >>>
        
        r_spx_sim = (mu_spx_next + sigma_t1_spx * z_spx) / 100
        r_eurusd_sim = (mu_eurusd_next + sigma_t1_eurusd * z_eurusd) / 100
        r_portfolio_sim = 0.5 * r_spx_sim + 0.5 * r_eurusd_sim
        
        var_99 = np.percentile(r_portfolio_sim, 1)
        es_99 = r_portfolio_sim[r_portfolio_sim <= var_99].mean()
        
        daily_forecasts[name] = {'VaR_99': var_99, 'ES_99': es_99}
        
    return daily_forecasts

# --- 3. Main execution block ---
if __name__ == '__main__':
    try:
        copula_input_data = pd.read_csv('copula_input_data.csv', index_col='Date', parse_dates=True).dropna()
        full_data = pd.read_csv('spx_eurusd_daily_data.csv', index_col='Date', parse_dates=True)
        
        copula_params = get_copula_parameters(copula_input_data)

        out_of_sample_start = '2020-01-01'
        forecast_dates = full_data.loc[out_of_sample_start:].index

        all_forecasts = []

        for day in tqdm(forecast_dates, desc="Rolling Forecast VaR/ES"):
            t_index = full_data.index[full_data.index.get_loc(day) - 1]
            forecasts = run_simulation_for_day(t_index, full_data, copula_params)
            
            flat_forecasts = {'Date': day}
            for model_name, values in forecasts.items():
                flat_forecasts[f'{model_name}_VaR_99'] = values['VaR_99']
                flat_forecasts[f'{model_name}_ES_99'] = values['ES_99']
            all_forecasts.append(flat_forecasts)

        forecasts_df = pd.DataFrame(all_forecasts).set_index('Date')
        
        forecast_output_file = 'forecast_results.csv'
        forecasts_df.to_csv(forecast_output_file)
        
        print(f"\nAll forecasts saved to '{forecast_output_file}'.")
        print(forecasts_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()