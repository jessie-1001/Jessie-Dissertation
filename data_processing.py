# =============================================================================
# SCRIPT 01: DATA ACQUISITION AND PROCESSING
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera # Import Jarque-Bera test

def data_processing_and_summary():
    """
    Main function to download, process, save, and summarize financial data.
    """
    # --- 1. Set up parameters (No changes here) ---
    tickers = ['^GSPC', 'EUR=X']
    start_date = '2007-01-01'
    end_date = '2025-06-01'
    output_file = 'spx_eurusd_daily_data.csv'

    # --- 2. Download historical price data (No changes here) ---
    print(f"Downloading daily data for {tickers} from {start_date} to {end_date}...")
    try:
        price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        print("Data download complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return

    # --- 3. Clean and process the data (No changes here) ---
    price_data.rename(columns={'^GSPC': 'SPX', 'EUR=X': 'EURUSD'}, inplace=True)
    price_data_aligned = price_data.dropna()
    print(f"Data aligned. Original rows: {len(price_data)}, Aligned rows: {len(price_data_aligned)}.")
    return_data_aligned = np.log(price_data_aligned / price_data_aligned.shift(1))
    return_data_aligned.rename(columns={'SPX': 'SPX_Return', 'EURUSD': 'EURUSD_Return'}, inplace=True)
    final_data = pd.concat([price_data_aligned, return_data_aligned], axis=1).dropna()
    final_data.to_csv(output_file)
    print(f"Cleaned data saved successfully to '{output_file}'.")

    # <<< START OF ADDED CODE FOR TABLE 4.1 >>>
    # --- 4. Generate and print Descriptive Statistics Table (for Dissertation Chapter 4.2) ---
    print("\n\n" + "="*80)
    print(">>> OUTPUT FOR DISSERTATION: TABLE 4.1 <<<")
    
    returns = final_data[['SPX_Return', 'EURUSD_Return']]
    desc_stats = returns.describe().T
    
    # Add Skewness, Kurtosis, and Jarque-Bera test results
    desc_stats['Skewness'] = returns.skew()
    desc_stats['Kurtosis'] = returns.kurtosis() # Pandas kurtosis is excess kurtosis
    jb_spx = jarque_bera(returns['SPX_Return'].dropna())
    jb_eurusd = jarque_bera(returns['EURUSD_Return'].dropna())
    desc_stats['Jarque-Bera'] = [jb_spx[0], jb_eurusd[0]]
    desc_stats['JB p-value'] = [jb_spx[1], jb_eurusd[1]]
    
    # Format the table for display
    desc_stats_formatted = desc_stats[['mean', 'std', 'min', 'max', 'Skewness', 'Kurtosis', 'Jarque-Bera', 'JB p-value']]
    desc_stats_formatted.columns = ['Mean', 'Std. Dev.', 'Min', 'Max', 'Skewness', 'Excess Kurtosis', 'Jarque-Bera', 'p-value']
    
    print("--- Descriptive Statistics of Daily Log-Returns ---")
    print(desc_stats_formatted.to_markdown(index=True, floatfmt=".4f"))
    print("="*80 + "\n")
    # <<< END OF ADDED CODE FOR TABLE 4.1 >>>

if __name__ == '__main__':
    data_processing_and_summary()