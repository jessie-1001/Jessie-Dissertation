# =============================================================================
# SCRIPT 01: DATA ACQUISITION AND PROCESSING (ENHANCED)
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis
import seaborn as sns

def data_processing_and_summary():
    """
    Enhanced function to download, process, save, and summarize financial data
    with additional data quality checks.
    """
    # --- 1. Set up parameters ---
    tickers = ['^GSPC', 'EUR=X']
    start_date = '2007-01-01'
    end_date = '2025-06-01'
    output_file = 'spx_eurusd_daily_data.csv'
    plot_dir = 'data_quality_plots/'
    import os
    os.makedirs(plot_dir, exist_ok=True)  # 创建目录保存图表

    # --- 2. Download historical price data ---
    print(f"Downloading daily data for {tickers} from {start_date} to {end_date}...")
    try:
        price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        print(f"Data download complete. Retrieved {len(price_data)} days of data.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return

    # --- 3. Clean and process the data ---
    price_data.rename(columns={'^GSPC': 'SPX', 'EUR=X': 'EURUSD'}, inplace=True)
    price_data_aligned = price_data.dropna()
    
    print(f"\nData alignment:")
    print(f"Original rows: {len(price_data)}, After alignment: {len(price_data_aligned)}")
    print(f"Removed {len(price_data) - len(price_data_aligned)} rows due to missing values.")
    
    # 计算对数收益率
    return_data = np.log(price_data_aligned / price_data_aligned.shift(1))
    return_data.rename(columns={'SPX': 'SPX_Return', 'EURUSD': 'EURUSD_Return'}, inplace=True)
    
    # 合并价格和收益率数据
    final_data = pd.concat([price_data_aligned, return_data], axis=1).dropna()
    print(f"Final dataset size: {len(final_data)} rows after removing leading NaN.")
    
    # 保存数据
    final_data.to_csv(output_file)
    print(f"\nCleaned data saved successfully to '{output_file}'.")

    # --- 4. Generate and print Descriptive Statistics Table (Table 4.1) ---
    print("\n\n" + "="*80)
    print(">>> OUTPUT FOR DISSERTATION: TABLE 4.1 <<<")
    
    returns = final_data[['SPX_Return', 'EURUSD_Return']]
    
    # 计算描述性统计量
    desc_stats = returns.describe().T
    
    # 添加偏度、超额峰度和Jarque-Bera检验结果
    desc_stats['Skewness'] = returns.skew()
    desc_stats['Kurtosis'] = returns.kurtosis()  # 注意：pandas的kurtosis计算的是超额峰度
    
    # 使用scipy计算峰度（与pandas不同）
    desc_stats['Kurtosis_scipy'] = [
        kurtosis(returns['SPX_Return'].dropna(), fisher=False),
        kurtosis(returns['EURUSD_Return'].dropna(), fisher=False)
    ]
    
    # Jarque-Bera检验
    jb_spx = jarque_bera(returns['SPX_Return'].dropna())
    jb_eurusd = jarque_bera(returns['EURUSD_Return'].dropna())
    desc_stats['Jarque-Bera'] = [jb_spx[0], jb_eurusd[0]]
    desc_stats['JB p-value'] = [jb_spx[1], jb_eurusd[1]]
    
    # 格式化表格显示
    desc_stats_formatted = desc_stats[[
        'mean', 'std', 'min', 'max', 'Skewness', 'Kurtosis', 
        'Kurtosis_scipy', 'Jarque-Bera', 'JB p-value'
    ]]
    desc_stats_formatted.columns = [
        'Mean', 'Std. Dev.', 'Min', 'Max', 'Skewness', 'Excess Kurtosis (pandas)', 
        'Kurtosis (scipy)', 'Jarque-Bera', 'p-value'
    ]
    
    print("\n--- Descriptive Statistics of Daily Log-Returns ---")
    print(desc_stats_formatted.to_markdown(index=True, floatfmt=".4f"))
    print("="*80 + "\n")

    # --- 5. Enhanced Data Quality Checks ---
    print("\n" + "="*80)
    print(">>> ENHANCED DATA QUALITY CHECKS <<<")
    
    # 5.1 极端值检测
    print("\n--- Extreme Value Detection ---")
    extreme_spx = final_data[(final_data['SPX_Return'] < -0.05) | (final_data['SPX_Return'] > 0.05)]
    extreme_eurusd = final_data[(final_data['EURUSD_Return'] < -0.03) | (final_data['EURUSD_Return'] > 0.03)]
    
    print(f"SPX extreme returns (<-5% or >5%): {len(extreme_spx)} events")
    print(f"EURUSD extreme returns (<-3% or >3%): {len(extreme_eurusd)} events")
    
    if not extreme_spx.empty:
        print("\nSPX extreme return dates:")
        print(extreme_spx[['SPX', 'SPX_Return']].sort_values('SPX_Return').to_markdown())
    
    if not extreme_eurusd.empty:
        print("\nEURUSD extreme return dates:")
        print(extreme_eurusd[['EURUSD', 'EURUSD_Return']].sort_values('EURUSD_Return').to_markdown())
    
    # 5.2 数据连续性检查
    print("\n--- Data Continuity Check ---")
    trading_days = final_data.asfreq('B').index  # 转为工作日频率
    date_diff = trading_days.to_series().diff().dt.days
    gap_days = date_diff[date_diff > 3]  # 忽略周末
        
    if not gap_days.empty:
        print(f"Data gaps detected ({len(gap_days)} gaps):")
        gap_summary = gap_days.value_counts().reset_index()
        gap_summary.columns = ['Gap Size (days)', 'Count']
        print(gap_summary.to_markdown(index=False))
        
        # 识别具体缺口日期
        gap_dates = []
        for idx, gap in gap_days.items():
            prev_date = final_data.index[final_data.index.get_loc(idx) - 1]
            gap_dates.append(f"{prev_date.date()} to {idx.date()} ({gap} days)")
        
        print("\nSpecific gap periods:")
        for gap in gap_dates[:5]:  # 只显示前5个缺口
            print(f" - {gap}")
        if len(gap_dates) > 5:
            print(f" - ... and {len(gap_dates)-5} more gaps")
    else:
        print("No significant data gaps detected (all consecutive trading days).")
    
    # 5.3 可视化分析
    print("\n--- Generating Data Quality Visualizations ---")
    plt.figure(figsize=(14, 10))
    
    # 价格序列图
    plt.subplot(2, 2, 1)
    final_data['SPX'].plot(title='S&P 500 Price Series', color='blue')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    final_data['EURUSD'].plot(title='EUR/USD Exchange Rate', color='green')
    plt.grid(True)
    
    # 收益率分布图
    plt.subplot(2, 2, 3)
    sns.histplot(final_data['SPX_Return'], kde=True, color='blue', bins=50)
    plt.title('S&P 500 Return Distribution')
    plt.axvline(x=0, color='red', linestyle='--')
    
    plt.subplot(2, 2, 4)
    sns.histplot(final_data['EURUSD_Return'], kde=True, color='green', bins=50)
    plt.title('EUR/USD Return Distribution')
    plt.axvline(x=0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}price_and_return_distributions.png")
    plt.close()
    print(f"- Saved price and return distributions to {plot_dir}price_and_return_distributions.png")
    
    # 极端事件标记图
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    final_data['SPX_Return'].plot(title='S&P 500 Daily Returns', color='blue', alpha=0.7)
    plt.scatter(extreme_spx.index, extreme_spx['SPX_Return'], color='red', s=30, label='Extreme Returns')
    plt.axhline(y=-0.05, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    final_data['EURUSD_Return'].plot(title='EUR/USD Daily Returns', color='green', alpha=0.7)
    plt.scatter(extreme_eurusd.index, extreme_eurusd['EURUSD_Return'], color='red', s=30, label='Extreme Returns')
    plt.axhline(y=-0.03, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.03, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}extreme_return_events.png")
    plt.close()
    print(f"- Saved extreme return events plot to {plot_dir}extreme_return_events.png")
    
    # 5.4 数据质量报告
    print("\n" + "="*80)
    print(">>> DATA QUALITY REPORT SUMMARY <<<")
    print(f"- Total observations: {len(final_data)}")
    print(f"- Date range: {final_data.index[0].date()} to {final_data.index[-1].date()}")
    print(f"- SPX missing values: {final_data['SPX'].isnull().sum()}")
    print(f"- EURUSD missing values: {final_data['EURUSD'].isnull().sum()}")
    print(f"- SPX extreme returns: {len(extreme_spx)} ({len(extreme_spx)/len(final_data)*100:.2f}%)")
    print(f"- EURUSD extreme returns: {len(extreme_eurusd)} ({len(extreme_eurusd)/len(final_data)*100:.2f}%)")
    print(f"- Data gaps: {len(gap_days)}")
    print("="*80 + "\n")

if __name__ == '__main__':
    data_processing_and_summary()