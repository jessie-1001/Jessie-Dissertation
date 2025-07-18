# data_utils.py

import yfinance as yf
import pandas as pd
import numpy as np

def fetch_price_data(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """
    Download closing price data for multiple tickers from Yahoo Finance.

    Parameters:
        tickers (dict): Dictionary of asset names and their ticker symbols.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Multi-column DataFrame of closing prices.
    """
    price_data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end)
        price_data[name] = df["Close"]
    return pd.concat(price_data, axis=1).dropna()


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from price data.

    Parameters:
        price_df (pd.DataFrame): DataFrame of asset prices.

    Returns:
        pd.DataFrame: Log returns DataFrame.
    """
    return np.log(price_df / price_df.shift(1)).dropna()


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame to CSV.

    Parameters:
        df (pd.DataFrame): DataFrame to be saved.
        filename (str): File name to save the CSV as.
    """
    df.to_csv(filename)
