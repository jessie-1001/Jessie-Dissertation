# =============================================================================
# SCRIPT 01: DATA ACQUISITION AND PROCESSING
#
# This script downloads daily price data for the S&P 500 and EUR/USD,
# cleans and aligns the data, calculates log returns, and saves the final
# dataset to a CSV file for subsequent analysis.
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_processing():
    """
    Main function to download, process, and save financial data.
    """
    # --- 1. Set up parameters ---
    tickers = ['^GSPC', 'EUR=X']
    start_date = '2007-01-01'
    end_date = '2025-06-01'
    output_file = 'spx_eurusd_daily_data.csv'

    # --- 2. Download historical price data ---
    print(f"Downloading daily data for {tickers} from {start_date} to {end_date}...")
    try:
        price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        print("Data download complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return

    # --- 3. Clean and process the data ---
    price_data.rename(columns={'^GSPC': 'SPX', 'EUR=X': 'EURUSD'}, inplace=True)

    # Handle non-common trading days by dropping rows with any missing values
    original_rows = len(price_data)
    price_data_aligned = price_data.dropna()
    cleaned_rows = len(price_data_aligned)
    print(f"Data aligned. Original rows: {original_rows}, Aligned rows: {cleaned_rows}.")

    # Calculate continuously compounded (log) returns
    return_data_aligned = np.log(price_data_aligned / price_data_aligned.shift(1))
    return_data_aligned.rename(columns={'SPX': 'SPX_Return', 'EURUSD': 'EURUSD_Return'}, inplace=True)

    # Combine price and return data into a single DataFrame
    final_data = pd.concat([price_data_aligned, return_data_aligned], axis=1)

    # Drop the first row which will have NaN values for returns
    final_data = final_data.dropna()

    # --- 4. Save and verify the data ---
    final_data.to_csv(output_file)
    print(f"Cleaned data saved successfully to '{output_file}'.")

    print("\n--- Data Preview (First 5 Rows) ---")
    print(final_data.head())
    print("\n--- Summary Statistics of Returns ---")
    print(final_data[['SPX_Return', 'EURUSD_Return']].describe())

    print("\nPlotting normalized price series for visual verification...")
    (final_data[['SPX', 'EURUSD']] / final_data[['SPX', 'EURUSD']].iloc[0] * 100).plot(
        figsize=(15, 7),
        title='Normalized Price Series (Start of Period = 100)',
        grid=True
    )
    plt.ylabel('Normalized Price')
    plt.show()

if __name__ == '__main__':
    data_processing()