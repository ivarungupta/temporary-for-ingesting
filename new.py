import requests
import pandas as pd
import numpy as np
import time
import pickle
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
api_key = "bEiVRux9rewQy16TXMPxDqBAQGIW8UBd"
base_url = "https://financialmodelingprep.com/api/v3"
start_date = "1996-01-01"
end_date = "2024-12-31"
output_dir = "sp500_constituents"
os.makedirs(output_dir, exist_ok=True)
endpoint = f"{base_url}/sp500_constituent"
params = {
    "apikey": api_key
}
response = requests.get(endpoint, params=params)
if response.status_code == 200:
    constituents = response.json()
    df = pd.DataFrame(constituents)
    current_date = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"sp500_constituents_{current_date}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved S&P 500 constituents to {output_file}")
    print(f"\nTotal companies in S&P 500: {len(df)}")
    print("\nFirst few companies:")
    print(df[['symbol', 'name', 'sector']].head())
else:
    print(f"Error fetching data: {response.status_code}")
df = pd.read_csv('sp500_constituents/sp500_constituents_20250511.csv')  
symbols = df['symbol'].tolist()
with open("sp500_symbols.pickle", "wb") as f:
    pickle.dump(symbols, f)
print(f"Saved {len(symbols)} symbols to sp500_symbols.pickle")
with open("sp500_symbols.pickle", "rb") as f:
    loaded_symbols = pickle.load(f)
print("Loaded symbols from pickle file:")
print(f"Total symbols: {len(loaded_symbols)}")
print("\nFirst 20 symbols:")
print(loaded_symbols[:20])
tickers_considered = loaded_symbols[:20]
print(tickers_considered)
output_dir = "stock_prices"
os.makedirs(output_dir, exist_ok=True)
insufficient_data_stocks = []
def fetch_historical_prices(symbol):
    endpoint = f"{base_url}/historical-price-full/{symbol}"
    params = {
        "from": start_date,
        "to": end_date,
        "apikey": api_key
    }
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            return df
    return None
for i, symbol in enumerate(tickers_considered):
    print(f'\n[{i+1}/{len(tickers_considered)}] Fetching close price for {symbol}...')
    df = fetch_historical_prices(symbol)
    if df is not None:
        csv_path = os.path.join(output_dir, f"{symbol}_prices.csv")
        df.to_csv(csv_path, index=False)
        if len(df) < 630: 
            insufficient_data_stocks.append(symbol)
            print(f"Warning: {symbol} has only {len(df)} data points")
        else:
            print(f"Saved {len(df)} rows for {symbol}")
    time.sleep(0.5)

print("\nStocks with insufficient data (less than 630 rows):")
for stock in insufficient_data_stocks:
    print(f"{stock}: {len(pd.read_csv(os.path.join(output_dir, f'{stock}_prices.csv')))} rows")
final_tickers = [ticker for ticker in tickers_considered if ticker not in insufficient_data_stocks]
with open("final_tickers.pickle", "wb") as f:
    pickle.dump(final_tickers, f)
print(f"\nOriginal number of tickers: {len(tickers_considered)}")
print(f"Number of tickers with insufficient data: {len(insufficient_data_stocks)}")
print(f"Final number of tickers: {len(final_tickers)}")
print(final_tickers)
print("\nSummary of saved files:")
print(f"Total CSV files created: {len(os.listdir(output_dir))}")
print(f"Files are saved in: {os.path.abspath(output_dir)}")
with open("final_tickers.pickle", "rb") as f:
    final_tickers = pickle.load(f)
all_data = []
for symbol in final_tickers:
    file_path = os.path.join("stock_prices", f"{symbol}_prices.csv")
    df = pd.read_csv(file_path)
    df['symbol'] = symbol
    all_data.append(df)
combined_df = pd.concat(all_data, ignore_index=True)
combined_df = combined_df.sort_values(['date', 'symbol'], ascending=[False, True])
columns = combined_df.columns.tolist()
columns.remove('symbol')
columns.remove('date')
new_column_order = ['date', 'symbol'] + columns
combined_df = combined_df[new_column_order].reset_index(drop=True)
print("\nSummary of the data:")
print(f"Total number of rows: {len(combined_df)}")
print(f"Number of unique stocks: {combined_df['symbol'].nunique()}")
print(f"Date range: from {combined_df['date'].min()} to {combined_df['date'].max()}")
print("First few rows of the combined data:")
print(combined_df.head())
combined_df = combined_df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
combined_df[combined_df["symbol"]=="WSM"].sort_values('date', ascending=False)
combined_df2 = combined_df.copy()
combined_df2['date'] = pd.to_datetime(combined_df2['date'])
combined_df2.set_index(['date', 'symbol'], inplace=True)
combined_df2.sort_index(ascending=[False, True], inplace=True)
display(combined_df2)
unique_tickers = combined_df2.index.get_level_values('symbol').unique()
print(len(unique_tickers))
"""### Factor Definitions & Calculations:
1. **Quality:**
   - `net_profit_to_total_operate_revenue_ttm`: Measures profitability relative to revenue.
   - `DECM`: Detection of Earnings Manipulation, signals potential earnings manipulation.
   - `roe_ttm`: Return on Equity, indicates profitability against equity.
   - `GMI`: Gross Margin Index, measures changes in gross margin.
   - `ACCA`: Accruals, indicating earnings quality.

2. **Value:**
   - `financial_liability`: Total liabilities as a measure of financial leverage.
   - `cash_flow_to_price_ratio`: Valuation metric using cash flow.
   - `market_cap`: Total market value of the company's outstanding shares.
   - `net_profit_ttm`: Total net profit over the trailing twelve months.
   - `EBIT`: Earnings before interest and taxes, a measure of profitability.

3. **Emotional:**
   - `VOL20`: 20-day price volatility.
   - `DAVOL20`: 20-day average trading volume.
   - `VOSC`: Volume Oscillator, indicating market sentiment.
   - `VMACD`: Volume MACD, momentum indicator using volume.
   - `ATR14`: 14-day Average True Range, measures volatility.

4. **Growth:**
   - `PEG`: Price/Earnings to Growth ratio.
   - `net_profit_growth_rate`: Year-over-year growth in net profit.
   - `operating_revenue_growth_rate`: Year-over-year growth in revenue.
   - `net_asset_growth_rate`: Growth in net assets.
   - `net_operate_cashflow_growth_rate`: Growth in operating cash flow.

5. **Risk:**
   - `Variance20`: 20-day return variance.
   - `sharpe_ratio_20`: 20-day Sharpe ratio.
   - `Kurtosis20`: 20-day kurtosis of returns.
   - `Skewness20`: 20-day skewness of returns.
   - `sharpe_ratio_60`: 60-day Sharpe ratio.

6. **Stock:**
   - `net_asset_per_share`: Net assets per share.
   - `net_operate_cash_flow_per_share`: Operating cash flow per share.
   - `eps_ttm`: Earnings per share over trailing twelve months.
   - `retained_earnings_per_share`: Retained earnings per share.
   - `cashflow_per_share_ttm`: Cash flow per share over trailing twelve months.

7. **Momentum:**
   - `ROC20`: 20-day rate of change.
   - `Volume1M`: One-month trading volume.
   - `TRIX10`: 10-day triple exponential moving average.
   - `Price1M`: One-month price change.
   - `PLRC12`: 12-day price momentum.

8. **Technical:**
   - `MAC20`: 20-day moving average convergence divergence.
   - `boll_down`: Bollinger Band lower band.
   - `boll_up`: Bollinger Band upper band.
   - `MFI14`: 14-day Money Flow Index.

9. **Style:**
   - `size`: Size of the company (market cap).
   - `beta`: Beta value of the stock.
   - `momentum`: Momentum factor of the stock.
   - `book_to_price_ratio`: Book-to-price ratio.
   - `liquidity`: Liquidity of the stock.
   - `growth`: Growth factor of the stock.

"""
factors = {
    'net_profit_to_total_revenue_ttm': 'Quality',
    'roe_ttm': 'Quality',
    'roa_ttm' : 'Quality',
    'GMI': 'Quality',
    'ACCA': 'Quality',
    'debt_to_asset_ratio' : 'Quality',

    'financial_liability': 'Value',
    'cash_flow_to_price_ratio_ttm': 'Value',
    'price_to_book_ratio' : 'Value',
    'price_to_sales_ratio_ttm' : 'Value',
    'price_to_earnings_ratio_ttm' : 'Value',
    'total_liability_to_total_asset_ratio' : 'Value',
    'net_profit_ttm': 'Value',
    'working_capital_ratio' : 'Value',
    'quick_ratio' : 'Value',
    'debt_to_equity_ratio' : 'Value',
    'operate_cash_flow_to_total_asset_ratio' : 'Value',
    'operate_cash_flow_to_total_liabilities_ratio' : 'Value',
    'operate_cash_flow_to_net_profit_ratio' : 'Value',
    'EV_to_operate_cash_flow_ratio' : 'Value',
    'debt_to_EBITDA_ratio' : 'Value',

    'EPS_growth_rate_ttm' : 'Growth',
    'PEG_ttm': 'Growth',
    'net_profit_growth_rate_ttm': 'Growth',
    'revenue_growth_rate_ttm': 'Growth',
    'net_asset_growth_rate': 'Growth',
    'operate_cash_flow_growth_rate_ttm': 'Growth',

    'net_asset_per_share': 'Stock',
    'net_operate_cash_flow_per_share_ttm': 'Stock',
    'retained_earnings_per_share': 'Stock',
    'market_cap(size)': 'Stock',

    'VOL60': 'Emotional',
    'DAVOL60': 'Emotional',
    'VOSC': 'Emotional',
    'VMACD': 'Emotional',
    'ATR42': 'Emotional',

    'ROC60': 'Momentum',
    'Volume1Q': 'Momentum',
    'TRIX30': 'Momentum',
    'Price1Q': 'Momentum',
    'PLRC36': 'Momentum',

    'Variance60': 'Risk',
    'Skewness60': 'Risk',
    'Kurtosis60': 'Risk',
    'sharpe_ratio_20': 'Risk',
    'sharpe_ratio_60': 'Risk',

    'MACD60': 'Technical',
    'boll_down': 'Technical',
    'boll_up': 'Technical',
    'MFI42': 'Technical',

    'growth': 'Style',
    'momentum': 'Style',
    'beta': 'Style',
    'liquidity': 'Style',
}
factor_list = list(factors.keys())
print("Total number of factors:", len(factor_list))
print("Factor list:", factor_list)
"""#FACTORS FROM OHLCV DATA - DAILY"""
combined_df3 = combined_df.reset_index()
combined_df3 = combined_df3.drop(combined_df3.columns[0], axis=1)
combined_df3['date'] = pd.to_datetime(combined_df3['date'])
combined_df3 = combined_df3.sort_values(['symbol', 'date'])
risk_free_rate_annual = 0.045
risk_free_rate_20 = (1 + risk_free_rate_annual) ** (20 / 252) - 1
risk_free_rate_60 = (1 + risk_free_rate_annual) ** (60 / 252) - 1
factor_columns = [
    'VOL60', 'DAVOL60', 'VOSC', 'VMACD', 'ATR42',
    'ROC60', 'Volume1Q', 'TRIX30', 'Price1Q', 'PLRC36',
    'Variance60', 'Skewness60', 'Kurtosis60', 'SharpeRatio20', 'SharpeRatio60',
    'MACD60', 'boll_up', 'boll_down', 'MFI42',
    'GrowthRate', 'Momentum', 'Beta'
]
for col in factor_columns:
    combined_df3[col] = np.nan
def calculate_volume_volatility_vol60(df, window=60):
    """Calculate 60-day volume volatility."""
    return df['close'].pct_change().rolling(window=window).std()
def calculate_volume_ma_davol60(df, window=60):
    """Calculate 60-day average volume."""
    return df['volume'].rolling(window=window).mean()
def calculate_volume_oscillator_vosc(df, window=60):
    """Calculate volume oscillator."""
    volume_ma = df['volume'].rolling(window=window).mean()
    return df['volume'] - volume_ma
def calculate_volume_macd(df, fast_span=36, slow_span=78):
    """Calculate volume MACD."""
    fast_ma = df['volume'].ewm(span=fast_span).mean()
    slow_ma = df['volume'].ewm(span=slow_span).mean()
    return fast_ma - slow_ma
def calculate_atr42(df, window=42):
    """Calculate 42-day Average True Range."""
    return (df['high'] - df['low']).rolling(window=window).mean()
def calculate_rate_of_change_roc60(df, window=60):
    """Calculate 60-day rate of change."""
    return (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
def calculate_volume_quarterly_1q(df, window=60):
    """Calculate quarterly volume."""
    return df['volume'].rolling(window=window).sum()
def calculate_trix30(df, span=30):
    """Calculate TRIX indicator."""
    triple_ema = df['close'].ewm(span=span).mean().ewm(span=span).mean().ewm(span=span).mean()
    return triple_ema.pct_change(periods=1)
def calculate_price_quarterly_price1q(df, window=60):
    """Calculate quarterly price change."""
    return df['close'] - df['close'].shift(window)
def calculate_price_level_ratio_PLRC36(df, window=36):
    """Calculate price level ratio."""
    return df['close'].rolling(window=window).mean() / df['close'].shift(window) - 1
def calculate_variance_60(df, window=60):
    """Calculate 60-day variance of returns."""
    return df['close'].pct_change().rolling(window=window).var()
def calculate_skewness60(df, window=60):
    """Calculate 60-day skewness of returns."""
    return df['close'].pct_change().rolling(window=window).skew()
def calculate_kurtosis60(df, window=60):
    """Calculate 60-day kurtosis of returns."""
    return df['close'].pct_change().rolling(window=window).kurt()
def calculate_sharpe_ratio(df, window, risk_free_rate):
    """Calculate Sharpe ratio for given window."""
    returns = df['close'].pct_change()
    annualized_returns = returns.rolling(window=window).mean() * 252
    annualized_std = returns.rolling(window=window).std() * np.sqrt(252)
    return (annualized_returns - risk_free_rate) / annualized_std
def calculate_macd_60(df, fast_span=36, slow_span=78):
    """Calculate MACD indicator."""
    fast_ma = df['close'].ewm(span=fast_span).mean()
    slow_ma = df['close'].ewm(span=slow_span).mean()
    return fast_ma - slow_ma
def calculate_bollinger_bands(df, window=60, num_std=2):
    """Calculate Bollinger Bands."""
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    return {
        'boll_up': rolling_mean + (rolling_std * num_std),
        'boll_down': rolling_mean - (rolling_std * num_std)
    }
def calculate_mfi_42(df, window=42):
    """Calculate Money Flow Index."""
    typical_price = (df['close'] + df['high'] + df['low']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_flow_sum = positive_flow.rolling(window=window).sum()
    negative_flow_sum = negative_flow.rolling(window=window).sum()
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    return 100 - (100 / (1 + money_flow_ratio))
def calculate_growth_rate(df, window=252):
    """Calculate growth rate using log returns."""
    return np.log(df['close'] / df['close'].shift(window))
def calculate_momentum(df, window=252):
    """Calculate momentum as price ratio."""
    return df['close'] / df['close'].shift(window)
symbols = combined_df3['symbol'].unique()
for symbol in symbols:
    try:
        stock_data = combined_df3[combined_df3['symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('date', ascending=True)
        stock_data['returns'] = stock_data['close'].pct_change()
        stock_data['VOL60'] = calculate_volume_volatility_vol60(stock_data)
        stock_data['DAVOL60'] = calculate_volume_ma_davol60(stock_data)
        stock_data['VOSC'] = calculate_volume_oscillator_vosc(stock_data)
        stock_data['VMACD'] = calculate_volume_macd(stock_data)
        stock_data['ATR42'] = calculate_atr42(stock_data)
        # Momentum Factors
        stock_data['ROC60'] = calculate_rate_of_change_roc60(stock_data)
        stock_data['Volume1Q'] = calculate_volume_quarterly_1q(stock_data)
        stock_data['TRIX30'] = calculate_trix30(stock_data)
        stock_data['Price1Q'] = calculate_price_quarterly_price1q(stock_data)
        stock_data['PLRC36'] = calculate_price_level_ratio_PLRC36(stock_data)
        # Risk Factors
        stock_data['Variance60'] = calculate_variance_60(stock_data)
        stock_data['Skewness60'] = calculate_skewness60(stock_data)
        stock_data['Kurtosis60'] = calculate_kurtosis60(stock_data)
        stock_data['SharpeRatio20'] = calculate_sharpe_ratio(stock_data, 20, risk_free_rate_20)
        stock_data['SharpeRatio60'] = calculate_sharpe_ratio(stock_data, 60, risk_free_rate_60)
        # Technical Factors
        stock_data['MACD60'] = calculate_macd_60(stock_data)
        bollinger_bands = calculate_bollinger_bands(stock_data)
        stock_data['boll_up'] = bollinger_bands['boll_up']
        stock_data['boll_down'] = bollinger_bands['boll_down']
        stock_data['MFI42'] = calculate_mfi_42(stock_data)
        # Style Factors
        stock_data['GrowthRate'] = calculate_growth_rate(stock_data)
        stock_data['Momentum'] = calculate_momentum(stock_data)
        # Forward fill any remaining NaN values (up to 5 days)
        for col in factor_columns:
            stock_data[col] = stock_data[col].ffill(limit=5)
        # Sort back to descending order (newest to oldest) before updating
        stock_data = stock_data.sort_values('date', ascending=False)
        # Update the original DataFrame with calculated values
        combined_df3.loc[combined_df3['symbol'] == symbol, factor_columns] = stock_data[factor_columns]
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        continue
# Sort back to descending order (newest to oldest) if needed
combined_df3 = combined_df3.sort_values(['symbol', 'date'], ascending=[True, False])
##Calculate Beta
# Sort combined_df3 from old date to new date
combined_df3 = combined_df3.sort_values(['symbol', 'date'], ascending=[True, True])
# Calculate stock returns for all stocks
combined_df3['stock_returns'] = combined_df3.groupby('symbol')['close'].pct_change()
# Get benchmark data and calculate returns
benchmark = '^GSPC'  # S&P 500 ETF benchmark
def get_benchmark_data():
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{benchmark}?serietype=line&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and "historical" in data:
            return [(entry["date"], entry["close"]) for entry in data["historical"] if start_date <= entry["date"] <= end_date]
    except Exception as e:
        print(f"Error fetching benchmark data: {str(e)}")
    return []
# Fetch benchmark data
benchmark_data = get_benchmark_data()
benchmark_df = pd.DataFrame(benchmark_data, columns=['date', 'close'])
benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
benchmark_df.set_index('date', inplace=True)
benchmark_df.sort_index(ascending=True, inplace=True)
# Calculate benchmark returns
benchmark_df['market_returns'] = benchmark_df['close'].pct_change()
# Merge benchmark returns with combined_df3
#drop any existing market_returns column if it exists
if 'market_returns' in combined_df3.columns:
    combined_df3 = combined_df3.drop('market_returns', axis=1)
combined_df3 = combined_df3.merge(
    benchmark_df['market_returns'].reset_index(),
    on='date',
    how='left'
)
# Calculate beta for all dates
window = 60  # 60-day rolling window
# Group by symbol and calculate rolling beta
for symbol in combined_df3['symbol'].unique():
    try:
        # Get data for current symbol
        symbol_data = combined_df3[combined_df3['symbol'] == symbol].copy()
        # Calculate rolling beta
        rolling_beta = (symbol_data['stock_returns'].rolling(window=window, min_periods=30)
                       .cov(symbol_data['market_returns']) /
                       symbol_data['market_returns'].rolling(window=window, min_periods=30).var())
        # Update beta values in combined_df3
        combined_df3.loc[combined_df3['symbol'] == symbol, 'Beta'] = rolling_beta.values
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        continue
# Forward fill any remaining NaN values (up to 5 days)
combined_df3['Beta'] = combined_df3.groupby('symbol')['Beta'].ffill(limit=5)
# Sort back to descending order (newest to oldest)
combined_df3 = combined_df3.sort_values(['symbol', 'date'], ascending=[True, False]).reset_index(drop=True)
print("\nFirst few rows of the combined DataFrame with factors:")
display(combined_df3.round(4))
print("\nNumber of NaN Values for each factor:")
print(combined_df3[factor_columns].isna().sum())
# Function to display NaN values for each stock
def display_stock_nan_values(df):
    """
    Display total NaN values for each stock.
    Args:
        df: DataFrame containing the factors
    """
    factor_columns = [
        'GrowthRate', 'Momentum', 'Beta',
        'VOL60', 'DAVOL60', 'VOSC', 'VMACD', 'ATR42',
        'Variance60', 'Skewness60', 'Kurtosis60', 'SharpeRatio20', 'SharpeRatio60',
        'ROC60', 'Volume1Q', 'TRIX30', 'Price1Q', 'PLRC36',
        'MACD60', 'boll_up', 'boll_down', 'MFI42'
    ]
    # Group by symbol and count NaN values
    nan_counts = df.groupby('symbol')[factor_columns].apply(lambda x: x.isna().sum())
    # Calculate total NaN values for each stock
    nan_counts['Total_NaN'] = nan_counts.sum(axis=1)
    # Sort by total NaN values in descending order
    nan_counts = nan_counts.sort_values('Total_NaN', ascending=False)
    # Display results
    print("\nNaN values for each stock:")
    display(nan_counts)
# Call the function
display_stock_nan_values(combined_df3)
daily_tech_factors = combined_df3[['symbol', 'date'] + factor_columns].reset_index(drop=True)
display(daily_tech_factors.round(4))
#daily_tech_factors.to_csv('daily_tech_factors.csv', index = False)
# Convert date to datetime if not already
daily_tech_factors['date'] = pd.to_datetime(daily_tech_factors['date'])
quarterly_dates = pd.date_range(start=start_date, end=end_date, freq='QE').strftime("%Y-%m-%d").tolist()
quarterly_dates = pd.to_datetime(quarterly_dates)
# Resample daily data to quarterly
quarterly_tech_factors = daily_tech_factors.copy()
quarterly_tech_factors['date'] = pd.to_datetime(quarterly_tech_factors['date'])
# Group by symbol and resample to quarterly frequency
quarterly_tech_factors = (quarterly_tech_factors
    .set_index(['symbol', 'date'])
    .groupby('symbol')
    .resample('QE', level='date')
    .last()
    .reset_index()
)
# Filter for only the dates in our quarterly_dates list
quarterly_tech_factors = quarterly_tech_factors[
    quarterly_tech_factors['date'].isin(quarterly_dates)
]
# Sort by symbol and date in descending order (latest first)
quarterly_tech_factors = quarterly_tech_factors.sort_values(['symbol', 'date'], ascending=[True, False])
# Drop the oldest quarter for each stock
quarterly_tech_factors = quarterly_tech_factors.groupby('symbol').apply(
    lambda x: x.iloc[:-1]).reset_index(drop=True)
print("\nDate range in quarterly technical factors:")
print("Earliest date:", quarterly_tech_factors['date'].min())
print("Latest date:", quarterly_tech_factors['date'].max())
print("\nFirst few rows of quarterly technical factors:")
display(quarterly_tech_factors.head().round(4))
print("\nMissing values in quarterly technical factors:")
display(quarterly_tech_factors.isna().sum())
print("\nQuarterly Technical Factors are saved to 'quarterly_tech_factors.csv'")
quarterly_tech_factors.to_csv('quarterly_tech_factors.csv', index = False)
quarterly_tech_factors
"""#FACTORS FROM FINANCIAL DATA - QUARTERLY"""
#Fetching Financial Statements
# Load final_tickers
with open("final_tickers.pickle", "rb") as f:
    final_tickers = pickle.load(f)
# Load combined_df2 to determine the date range (if needed)
# Assuming combined_df2 already exists
start_date = combined_df2.index.get_level_values('date').min().strftime('%Y-%m-%d')
end_date = combined_df2.index.get_level_values('date').max().strftime('%Y-%m-%d')
print(f"Fetching financials from {start_date} to {end_date}")
# Create directory to store individual financial statements as CSV
os.makedirs("financials_csv", exist_ok=True)
# Function to fetch financial data for a symbol
def fetch_financial_data(symbol, statement_type):
    url = f"{base_url}/{statement_type}/{symbol}"
    params = {
        "period": "quarter",
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        if not df.empty:
            df['symbol'] = symbol
            df['statement_type'] = statement_type
        return df
    else:
        print(f"Failed to fetch {statement_type} for {symbol} | Status Code: {response.status_code}")
        return pd.DataFrame()
# Main loop to fetch all financials and save to CSV
financial_data = {}
for i, symbol in enumerate(final_tickers):
    print(f"\n[{i+1}/{len(final_tickers)}] Fetching financials for {symbol}...")
    income_df = fetch_financial_data(symbol, "income-statement")
    balance_df = fetch_financial_data(symbol, "balance-sheet-statement")
    cashflow_df = fetch_financial_data(symbol, "cash-flow-statement")
    # Optional: Filter by date range if needed
    for merged_df2_with_ttm in [income_df, balance_df, cashflow_df]:
        if not merged_df2_with_ttm.empty and 'date' in merged_df2_with_ttm.columns:
            merged_df2_with_ttm['date'] = pd.to_datetime(merged_df2_with_ttm['date'])
            merged_df2_with_ttm.dropna(subset=['date'], inplace=True)
    income_df = income_df[income_df['date'].between(start_date, end_date)]
    balance_df = balance_df[balance_df['date'].between(start_date, end_date)]
    cashflow_df = cashflow_df[cashflow_df['date'].between(start_date, end_date)]
    # Save CSVs
    if not income_df.empty:
        income_df.to_csv(f"financials_csv/{symbol}_income_statement.csv", index=False)
    if not balance_df.empty:
        balance_df.to_csv(f"financials_csv/{symbol}_balance_sheet.csv", index=False)
    if not cashflow_df.empty:
        cashflow_df.to_csv(f"financials_csv/{symbol}_cash_flow_statement.csv", index=False)

    # Save to dictionary
    financial_data[symbol] = {
        "income-statement": income_df,
        "balance-sheet-statement": balance_df,
        "cash-flow-statement": cashflow_df
    }

    time.sleep(0.5)  # API rate limit

# # Optionally, save the financial data to a pickle file as well
# with open("financial_statements_quarterly.pickle", "wb") as f:
#     pickle.dump(financial_data, f)

#print("\nAll financial statements saved to 'financials_csv' and 'financial_statements_quarterly.pickle'.")
print("\nAll financial statements saved to 'financials_csv'.")

def print_financial_statement(symbol, statement_type, start_date=None, end_date=None):
    """
    Print financial statement for a given stock and period.

    Args:
        symbol (str): Stock symbol
        statement_type (str): Type of statement ('income', 'balance', or 'cashflow')
        start_date (str): Start date in 'YYYY-MM-DD' format (optional)
        end_date (str): End date in 'YYYY-MM-DD' format (optional)
    """
    # Validate statement type
    valid_statements = {
        "income": "income_statement",
        "balance": "balance_sheet",
        "cashflow": "cash_flow_statement"
    }

    if statement_type.lower() not in valid_statements:
        print(f"❌ Invalid statement type. Please choose from: {', '.join(valid_statements.keys())}")
        return

    # Build file path
    file_path = f"financials_csv/{symbol}_{valid_statements[statement_type.lower()]}.csv"

    try:
        # Load the CSV
        df = pd.read_csv(file_path, parse_dates=["date"])

        # Print basic information
        print(f"\n📊 {statement_type.upper()} Statement for {symbol}")
        print(f"📅 Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"📄 Total number of periods: {len(df)}")

        # Filter by date if provided
        if start_date or end_date:
            if start_date:
                df = df[df["date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["date"] <= pd.to_datetime(end_date)]
            print(f"📅 Filtered period: {start_date or 'beginning'} to {end_date or 'end'}")

        # Print the data
        #print(f"\n�� Statement data:")
        display(df.sort_values('date', ascending=False))

        print(f"\n📋 Available columns:")
        print(df.columns.tolist())

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please check if the stock symbol is correct and the file exists.")

    except Exception as e:
        print(f"⚠️ Error: {e}")


# Example usage:
print_financial_statement("APO", "income", "2023-01-01", "2024-12-31")

def load_financial_statements(csv_dir="financials_csv"):
    """
    Load and combine financial statements for all stocks into three separate dataframes.

    Args:
        csv_dir (str): Directory containing financial statement CSV files

    Returns:
        tuple: (income_df_all, balance_df_all, cashflow_df_all) containing combined financial statements
    """
    # Predefine the selected columns
    balance_cols = ['date', 'symbol', 'reportedCurrency', 'cik', 'totalAssets', 'totalCurrentAssets',
                    'totalLiabilities', 'totalCurrentLiabilities', 'cashAndCashEquivalents',
                    'inventory', 'totalDebt', 'retainedEarnings', 'totalStockholdersEquity']

    income_cols = ['date', 'symbol', 'reportedCurrency', 'cik', 'revenue', 'grossProfit', 'netIncome', 'eps',
                   'ebitda', 'weightedAverageShsOut']

    cashflow_cols = ['date', 'symbol', 'reportedCurrency', 'cik', 'operatingCashFlow']

    # Initialize empty lists to collect data
    income_dfs = []
    balance_dfs = []
    cashflow_dfs = []

    # Get list of all available stock symbols from the directory
    available_symbols = set()
    for file in os.listdir(csv_dir):
        if file.endswith('_income_statement.csv'):
            symbol = file.split('_')[0]
            available_symbols.add(symbol)

    print(f"Found {len(available_symbols)} stocks with financial statements")

    # Loop through all available tickers and read their statements
    for symbol in sorted(available_symbols):
        try:
            # Load income statement
            income_path = os.path.join(csv_dir, f"{symbol}_income_statement.csv")
            if os.path.exists(income_path):
                df_income = pd.read_csv(income_path, parse_dates=["date"])
                df_income = df_income[income_cols]
                income_dfs.append(df_income)

            # Load balance sheet
            balance_path = os.path.join(csv_dir, f"{symbol}_balance_sheet.csv")
            if os.path.exists(balance_path):
                df_balance = pd.read_csv(balance_path, parse_dates=["date"])
                df_balance = df_balance[balance_cols]
                balance_dfs.append(df_balance)

            # Load cash flow statement
            cashflow_path = os.path.join(csv_dir, f"{symbol}_cash_flow_statement.csv")
            if os.path.exists(cashflow_path):
                df_cashflow = pd.read_csv(cashflow_path, parse_dates=["date"])
                df_cashflow = df_cashflow[cashflow_cols]
                cashflow_dfs.append(df_cashflow)

        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}")

    # Concatenate dataframes
    income_df_all = pd.concat(income_dfs, ignore_index=True)
    balance_df_all = pd.concat(balance_dfs, ignore_index=True)
    cashflow_df_all = pd.concat(cashflow_dfs, ignore_index=True)

    # Sort all dataframes by date and symbol
    for df in [income_df_all, balance_df_all, cashflow_df_all]:
        df.sort_values(['date', 'symbol'], ascending=[False, True], inplace=True)

    # Print summary information
    # print("\nCombined Financial Statements Summary:")
    # print(f"Income Statement: {income_df_all.shape[0]} rows, {income_df_all.shape[1]} columns")
    # print(f"Balance Sheet: {balance_df_all.shape[0]} rows, {balance_df_all.shape[1]} columns")
    # print(f"Cash Flow Statement: {cashflow_df_all.shape[0]} rows, {cashflow_df_all.shape[1]} columns")
    # print(f"Income DF shape: {income_df_all.shape}")
    # print(f"Balance DF shape: {balance_df_all.shape}")
    # print(f"Cashflow DF shape: {cashflow_df_all.shape}")

    print("\nDate Range:")
    print(f"From: {min(income_df_all['date'].min(), balance_df_all['date'].min(), cashflow_df_all['date'].min())}")
    print(f"To: {max(income_df_all['date'].max(), balance_df_all['date'].max(), cashflow_df_all['date'].max())}")

    print("\nNumber of unique stocks in each statement:")
    print(f"Income Statement: {income_df_all['symbol'].nunique()}")
    print(f"Balance Sheet: {balance_df_all['symbol'].nunique()}")
    print(f"Cash Flow Statement: {cashflow_df_all['symbol'].nunique()}")

    return income_df_all, balance_df_all, cashflow_df_all

# Load the financial statements
income_df_all, balance_df_all, cashflow_df_all = load_financial_statements()

# Optional: Save the combined dataframes to CSV files
# income_df.to_csv('combined_income_statements.csv', index=False)
# balance_df.to_csv('combined_balance_sheets.csv', index=False)
# cashflow_df.to_csv('combined_cashflow_statements.csv', index=False)

income_df_all.columns

# Merge the dataframes
merged_df = income_df_all.merge(balance_df_all, on=["date", "symbol", "reportedCurrency", "cik"], how="inner")  # Using inner join to ensure we only keep rows with data in all statements

merged_df = merged_df.merge(cashflow_df_all, on=["date", "symbol", "reportedCurrency", "cik"], how="inner")

# Sort the merged dataframe
merged_df = merged_df.sort_values(['date', 'symbol'], ascending=[False, True])

# Print information about the merged dataframe
print("Merged Financial Statements Summary:")
print(f"Total number of rows: {len(merged_df)}")
print(f"Total number of columns: {len(merged_df.columns)}")
print(f"Number of unique stocks: {merged_df['symbol'].nunique()}")
#print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")

#Save to CSV
merged_df.to_csv('merged_financial_data.csv', index=False)

print("\nFirst few rows of merged dataframe:")
merged_df.head()

merged_df.columns

def calculate_ttm(df, columns):
    """
    Calculate TTM (Trailing Twelve Months) for specified columns in the DataFrame.
    TTM is computed by summing the values of the most recent quarter + the previous 3 quarters.

    Args:
        df (pd.DataFrame): DataFrame containing financial data
        columns (list): List of column names to calculate TTM for

    Returns:
        pd.DataFrame: DataFrame with TTM columns added
    """
    # Create a copy to avoid modifying the original dataframe
    df_ttm = df.copy()

    # Verify all columns exist in the dataframe
    missing_cols = [col for col in columns if col not in df_ttm.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")

    # Calculate TTM for each column
    for column in columns:
        ttm_values = []
        ttm_column = f'{column}_ttm'

        for symbol, group in df_ttm.groupby('symbol'):
            # Sort by date descending to ensure most recent is first
            group = group.sort_values('date', ascending=False)
            ttm = [None] * len(group)

            # Calculate TTM for each quarter
            for i in range(len(group) - 3):
                # Sum the latest quarter + previous 3 quarters
                ttm_sum = group.iloc[i:i+4][column].sum()
                ttm[i] = ttm_sum

            # Assign TTM values back to the dataframe
            df_ttm.loc[group.index, ttm_column] = ttm

    return df_ttm

# Columns to calculate TTM for
ttm_columns = [
    'netIncome',
    'operatingCashFlow',
    'revenue',
    'grossProfit',
    'eps',
    'ebitda'
]

# Create list of TTM column names
ttm_columns_ttm = [f"{col}_ttm" for col in ttm_columns]

try:
    # Apply TTM calculation
    merged_df_with_ttm = calculate_ttm(merged_df, ttm_columns)

    # Drop rows where any of the TTM columns are NA
    merged_df_with_ttm = merged_df_with_ttm.dropna(subset=ttm_columns_ttm)

    # Print summary information
    print("\nTTM Calculation Summary:")
    print(f"Original number of rows: {len(merged_df)}")
    print(f"Rows after TTM calculation: {len(merged_df_with_ttm)}")
    print(f"Number of rows removed: {len(merged_df) - len(merged_df_with_ttm)}")
    print(f"Number of unique stocks: {merged_df_with_ttm['symbol'].nunique()}")
    print(f"Date range: {merged_df_with_ttm['date'].min()} to {merged_df_with_ttm['date'].max()}")

    # Print TTM columns added
    print("\nTTM columns added:")
    for col in ttm_columns_ttm:
        print(f"- {col}")

    # Save to CSV
    merged_df_with_ttm.to_csv('merged_financial_data_with_ttm.csv', index=False)

    # Display sample data
    print("\nSample data:")
    display(merged_df_with_ttm)

except Exception as e:
    print(f"Error during TTM calculation: {str(e)}")

merged_df_with_ttm.columns

len(merged_df_with_ttm['symbol'].unique())

def get_previous_price_data(df_price, target_date, symbol, max_lookback=5):
    """
    Get the most recent price data before or on the target date for a stock.

    Args:
        df_price (pd.DataFrame): DataFrame containing price data
        target_date (pd.Timestamp): Target date to find price for
        symbol (str): Stock symbol
        max_lookback (int): Maximum number of days to look back

    Returns:
        pd.Series: Price data (open, high, low, close, volume) or None if not found
    """
    # Reset index to make date and symbol regular columns
    df_price_reset = df_price.reset_index()

    # Get price data for the symbol up to target date
    symbol_data = df_price_reset[df_price_reset['symbol'] == symbol]
    symbol_data = symbol_data[symbol_data['date'] <= target_date]

    if not symbol_data.empty:
        # Sort by date descending to get most recent first
        symbol_data = symbol_data.sort_values('date', ascending=False)

        # Look back up to max_lookback days
        for i in range(min(max_lookback, len(symbol_data))):
            price_data = symbol_data.iloc[i]
            if not price_data[['open', 'high', 'low', 'close', 'volume']].isna().any():
                return price_data[['open', 'high', 'low', 'close', 'volume']]

    return None

# Create a copy of merged_df_with_ttm to work with
merged_df_with_ttm_and_price = merged_df_with_ttm.copy()

# Initialize price columns
price_columns = ['open', 'high', 'low', 'close', 'volume']
for col in price_columns:
    merged_df_with_ttm_and_price[col] = None

# Get unique dates and symbols from merged_df_with_ttm
unique_dates = merged_df_with_ttm_and_price['date'].unique()
unique_symbols = merged_df_with_ttm_and_price['symbol'].unique()

# For each date and symbol combination
for date in unique_dates:
    for symbol in unique_symbols:
        # Get the row index for this date and symbol
        mask = (merged_df_with_ttm_and_price['date'] == date) & (merged_df_with_ttm_and_price['symbol'] == symbol)
        if mask.any():
            # Get previous price data
            price_data = get_previous_price_data(combined_df2, date, symbol)

            if price_data is not None:
                # Update the price columns
                for col in price_columns:
                    merged_df_with_ttm_and_price.loc[mask, col] = price_data[col]

# Print summary information
print("\nMerged Data Summary:")
print(f"Total rows: {len(merged_df_with_ttm_and_price)}")
print(f"Rows with price data: {merged_df_with_ttm_and_price[price_columns].notna().all(axis=1).sum()}")
print(f"Rows missing price data: {merged_df_with_ttm_and_price[price_columns].isna().any(axis=1).sum()}")
# Print date range
print("\nDate range:")
print(f"From: {merged_df_with_ttm_and_price['date'].min()}")
print(f"To: {merged_df_with_ttm_and_price['date'].max()}")

# Display sample data
print("\nSample data:")
display(merged_df_with_ttm_and_price)

# Save to CSV
merged_df_with_ttm_and_price.to_csv('merged_financial_data_with_ttm_and_price.csv', index=False)

# Find rows where close price is missing in merged_with_prices
missing_close_prices = merged_df_with_ttm_and_price[merged_df_with_ttm_and_price['close'].isna()]
print(f"Total number of rows with missing close prices: {len(missing_close_prices)}")
print("\nStocks with missing close prices:")
display(missing_close_prices)

def check_stock_price(symbol, date):
    """
    Check the close price of a stock on a specific date.

    Args:
        symbol (str): Stock symbol
        date (str): Date in 'YYYY-MM-DD' format
    """
    # Convert date to datetime
    check_date = pd.to_datetime(date)

    # Get price data for the stock
    stock_data = combined_df2.xs(symbol, level='symbol')

    # Try to get the exact date
    try:
        price = stock_data.loc[check_date, 'close']
        print(f"\nClose price for {symbol} on {date}:")
        print(f"Exact date match: ${price:.2f}")
    except KeyError:
        # If exact date not found, get the most recent previous date
        try:
            previous_dates = stock_data.index[stock_data.index <= check_date]
            if len(previous_dates) > 0:
                most_recent_date = previous_dates[-1]
                price = stock_data.loc[most_recent_date, 'close']
                print(f"\nClose price for {symbol} on {date}:")
                print(f"No exact date match found.")
            else:
                print(f"\nNo price data available for {symbol} on or before {date}")
        except Exception as e:
            print(f"\nError finding price data: {str(e)}")

# Example usage:
# Check price for a specific stock and date
symbol = "EXE"
date = "2020-12-31"
check_stock_price(symbol, date)

# Drop rows with missing close prices
merged_df_with_ttm_and_price = merged_df_with_ttm_and_price.dropna(subset=['close'])

# Print confirmation of the cleaning
print(f"Rows dropped: {len(missing_close_prices)}")
print(f"Remaining rows: {len(merged_df_with_ttm_and_price)}")

print(merged_df_with_ttm_and_price['symbol'].nunique()) #no of unique tickers
#len(merged_price_financial['symbol'].unique())

print(len(merged_df_with_ttm_and_price.columns)) #no of columns

merged_df_with_ttm_and_price

merged_df_with_ttm_and_price.columns

"""## Calculation of Factors

In this step, we calculate various financial factors for each ticker using data from the income statement, balance sheet, and cash flow statements. These factors are calculated on a yearly basis and assigned to each day of the corresponding year. Below are the detailed explanations of each factor:

### 1. Quality Factors

- **Net Profit to Total Operating Revenue (TTM)**:
  This ratio is calculated as:
  $$
  \text{Net Profit to Total Operating Revenue (TTM)} = \frac{\text{Net Income}}{\text{Total Revenue}}
  $$
  This measures the profitability of the company relative to its total revenue.

- **Return on Equity (ROE) (TTM)**:
  This ratio is calculated as:
  $$
  \text{ROE (TTM)} = \frac{\text{Net Income}}{\text{Stockholders' Equity}}
  $$
  It indicates how effectively the company is using its equity base to generate profits.

- **Gross Margin Improvement (GMI)**:
  GMI is calculated as the change in gross margin compared to the previous year:
  $$
  \text{GMI} = \left( \frac{\text{Gross Profit}}{\text{Total Revenue}} \right)_{\text{current year}} - \left( \frac{\text{Gross Profit}}{\text{Total Revenue}} \right)_{\text{previous year}}
  $$
  This measures the improvement or decline in gross margin over the previous year.

- **Earnings Manipulation Detection (DECM)**:
  This factor detects potential earnings manipulation by analyzing changes in accounts receivable and inventory relative to total assets:
  $$
  \text{DECM} = \frac{\Delta \text{Accounts Receivable} + \Delta \text{Inventory}}{\text{Total Assets}}
  $$
  It signals potential earnings manipulation if there are significant increases in these components relative to the assets.

- **Accruals Quality Factor (ACCA)**:
  This factor measures the quality of earnings by comparing accruals to actual cash flows:
  $$
  \text{ACCA} = \frac{\text{Net Income} - \text{Cash Flow from Operating Activities}}{\text{Total Assets}}
  $$
  A higher value indicates lower quality of earnings.

### 2. Fundamental Factors

- **Total Liabilities**:
  This factor is directly taken from the balance sheet as:
  $$
  \text{Total Liabilities} = \text{Total Liabilities (Net Minority Interest)}
  $$
  It represents the total amount of liabilities the company holds.

- **Cash Flow to Price Ratio**:
  This ratio is calculated as:
  $$
  \text{Cash Flow to Price Ratio} = \frac{\text{Cash Flow from Continuing Operating Activities} / \text{Shares Outstanding}}{\text{Close Price}}
  $$
  It measures the cash flow generated by the company relative to its stock price.

- **Net Profit (TTM)**:
  This factor represents the net income of the company for the trailing twelve months (TTM):
  $$
  \text{Net Profit (TTM)} = \text{Net Income}
  $$

- **Earnings Before Interest and Taxes (EBIT)**:
  This factor represents the company's earnings before interest and taxes for the year:
  $$
  \text{EBIT} = \text{Earnings Before Interest and Taxes}
  $$

### 3. Growth Factors

- **Net Profit Growth Rate**:
  This factor measures the growth rate of net profit over the previous year:
  $$
  \text{Net Profit Growth Rate} = \frac{\text{Net Income}_{\text{current year}}}{\text{Net Income}_{\text{previous year}}} - 1
  $$

- **Operating Revenue Growth Rate**:
  This factor measures the growth rate of total revenue over the previous year:
  $$
  \text{Operating Revenue Growth Rate} = \frac{\text{Total Revenue}_{\text{current year}}}{\text{Total Revenue}_{\text{previous year}}} - 1
  $$

- **Net Asset Growth Rate**:
  This factor measures the growth rate of stockholders' equity over the previous year:
  $$
  \text{Net Asset Growth Rate} = \frac{\text{Stockholders' Equity}_{\text{current year}}}{\text{Stockholders' Equity}_{\text{previous year}}} - 1
  $$

- **Net Operating Cash Flow Growth Rate**:
  This factor measures the growth rate of cash flow from operating activities over the previous year:
  $$
  \text{Net Operating Cash Flow Growth Rate} = \frac{\text{Cash Flow from Continuing Operating Activities}_{\text{current year}}}{\text{Cash Flow from Continuing Operating Activities}_{\text{previous year}}} - 1
  $$

- **PEG Ratio**:
  The PEG ratio evaluates the stock's valuation by comparing the price-to-earnings (P/E) ratio with the net profit growth rate.
  $$
  \text{PEG} = \frac{\text{P/E Ratio}}{\text{Net Profit Growth Rate}}
  $$
  Here, the P/E Ratio is the ratio of the close price to earnings per share (EPS).

### 4. Stock Factors

- **Net Asset Per Share**:
  This factor is calculated as:
  $$
  \text{Net Asset Per Share} = \frac{\text{Stockholders' Equity}}{\text{Shares Outstanding}}
  $$
  It measures the equity per share held by the company.

- **Net Operating Cash Flow Per Share**:
  This factor is calculated using operating cash flow:
  $$
  \text{Net Operating Cash Flow Per Share} = \frac{\text{Operating Cash Flow}}{\text{Shares Outstanding}}
  $$

- **Cash Flow Per Share (TTM)**:
  This factor is calculated using free cash flow:
  $$
  \text{Cash Flow Per Share (TTM)} = \frac{\text{Free Cash Flow}}{\text{Shares Outstanding}}
  $$

- **Earnings Per Share (TTM)**:
  This factor is calculated as:
  $$
  \text{Earnings Per Share (TTM)} = \frac{\text{Net Income}}{\text{Shares Outstanding}}
  $$

- **Retained Earnings Per Share**:
  This factor is calculated as:
  $$
  \text{Retained Earnings Per Share} = \frac{\text{Retained Earnings}}{\text{Shares Outstanding}}
  $$

### 5. Style Factors

- **Book to Price Ratio**:
  If `Book Value Per Share` is available, it is calculated as:
  $$
  \text{Book to Price Ratio} = \frac{\text{Book Value Per Share}}{\text{Close Price}}
  $$
  If `Book Value Per Share` is not available, the calculation is done as:
  $$
  \text{Book to Price Ratio} = \frac{\text{Common Stock Equity}}{\text{Close Price}}
  $$
  This factor measures how the book value of the company compares to its market price.

- **Size (Market Capitalization)**:
  This factor is calculated as the product of the company's closing price and the number of shares outstanding. It represents the total market value of the company.
  $$
  \text{Market Capitalization} = \text{Close Price} \times \text{Shares Outstanding}
  $$
  This factor was calculated in Step 1 and stored as the `market_cap` column. It is also assigned to the `size` column for analysis.

- **Beta**:
  Beta measures the sensitivity of the stock's returns relative to the S&P 500 index. It is calculated as the covariance of the stock's returns with the market returns, divided by the variance of the market returns. A beta greater than 1 indicates higher sensitivity, while a beta less than 1 indicates lower sensitivity.
  $$
  \beta = \frac{\text{Cov}(\text{R}_{\text{stock}}, \text{R}_{\text{market}})}{\text{Var}(\text{R}_{\text{market}})}
  $$
  where $\text{R}_{\text{stock}}$ represents the daily returns of the stock and $\text{R}_{\text{market}}$ represents the daily returns of the S&P 500 index.

- **Liquidity**:
  Liquidity measures how easily the stock can be traded in the market without impacting its price. It is calculated as the ratio of the trading volume to the number of shares outstanding.
  $$
  \text{Liquidity} = \frac{\text{Volume}}{\text{Shares Outstanding}}
  $$
  Higher liquidity values indicate that the stock is more easily tradable in the market.

- **Momentum**:
  Momentum is calculated as the ratio of the current stock price to the stock price from 12 months ago. It captures the relative change in the stock price over the past year:
  $$
  \text{Momentum} = \frac{\text{Close}_{\text{today}}}{\text{Close}_{\text{252 days ago}}}
  $$
  where $\text{Close}_{\text{252 days ago}}$ is the closing price 252 trading days ago (approximately 1 year). A momentum value greater than 1 indicates that the stock has appreciated over the past year, while a value less than 1 indicates depreciation.

- **Growth**:
  Growth represents the logarithmic growth of the stock price over a 12-month period. It is calculated as the natural logarithm of the ratio between the current stock price and the stock price from 12 months ago:
  $$
  \text{Growth} = \ln \left( \frac{\text{Close}_{\text{today}}}{\text{Close}_{\text{252 days ago}}} \right)
  $$
  This is equivalent to the difference in the natural logarithm of the current stock price and the stock price from 252 trading days ago:
  $$
  \text{Growth} = \ln(\text{Close}_{\text{today}}) - \ln(\text{Close}_{\text{252 days ago}})
  $$
  This factor highlights the exponential growth or decline in the stock's value over the past year.

### 6. Emotional Factors

- **20-Day Volatility (VOL20)**:
  This factor measures the standard deviation of daily returns over the past 20 days, indicating the level of price variability. A higher value represents higher volatility and uncertainty.
  $$
  \text{VOL20} = \sqrt{\frac{\sum_{i=1}^{20} (\text{R}_{i} - \overline{\text{R}})^2}{20}}
  $$
  where $\text{R}_{i}$ represents daily returns and $\overline{\text{R}}$ is the average return over the past 20 days.

- **20-Day Average Volume (DAVOL20)**:
  This factor represents the average trading volume over the past 20 days, reflecting investor interest and liquidity in the stock.
  $$
  \text{DAVOL20} = \frac{\sum_{i=1}^{20} \text{Volume}_{i}}{20}
  $$

- **Volume Oscillator (VOSC)**:
  The volume oscillator calculates the difference between the current volume and its 20-day moving average, providing insight into changes in trading activity.
  $$
  \text{VOSC} = \text{Volume} - \text{DAVOL20}
  $$

- **Volume MACD (VMACD)**:
  This factor is the difference between two exponentially weighted moving averages (12-day and 26-day) of volume, signaling volume momentum.
  $$
  \text{VMACD} = \text{EMA}_{12}(\text{Volume}) - \text{EMA}_{26}(\text{Volume})
  $$

- **Average True Range (ATR14)**:
  ATR14 measures the average range between the high and low prices over the past 14 days, indicating market volatility and potential price movement.
  $$
  \text{ATR14} = \frac{\sum_{i=1}^{14} (\text{High}_{i} - \text{Low}_{i})}{14}
  $$

### 7. Risk Factors

- **20-Day Variance (Variance20)**:
  This factor measures the variance of daily returns over the past 20 days, providing insight into the dispersion of returns.
  $$
  \text{Variance20} = \frac{\sum_{i=1}^{20} (\text{R}_{i} - \overline{\text{R}})^2}{20}
  $$

- **20-Day Sharpe Ratio (sharpe_ratio_20)**:
  The Sharpe ratio measures the excess return per unit of risk (standard deviation) over the past 20 days.
  $$
  \text{Sharpe Ratio}_{20} = \frac{\overline{\text{R}} - \text{R}_{f}}{\text{VOL20}}
  $$
  where $\overline{\text{R}}$ is the average daily return, and $\text{R}_{f}$ is the risk-free rate (assumed to be zero for simplification).

- **20-Day Kurtosis (Kurtosis20)**:
  Kurtosis measures the "tailedness" of the return distribution over the past 20 days. A high value indicates more outliers.
  $$
  \text{Kurtosis}_{20} = \frac{\frac{1}{20} \sum_{i=1}^{20} (\text{R}_{i} - \overline{\text{R}})^4}{\left(\frac{1}{20} \sum_{i=1}^{20} (\text{R}_{i} - \overline{\text{R}})^2\right)^2}
  $$

- **20-Day Skewness (Skewness20)**:
  Skewness measures the asymmetry of the return distribution over the past 20 days. A negative skew indicates a higher probability of negative returns.
  $$
  \text{Skewness}_{20} = \frac{\frac{1}{20} \sum_{i=1}^{20} (\text{R}_{i} - \overline{\text{R}})^3}{\left(\frac{1}{20} \sum_{i=1}^{20} (\text{R}_{i} - \overline{\text{R}})^2\right)^{3/2}}
  $$

- **60-Day Sharpe Ratio (sharpe_ratio_60)**:
  This factor extends the Sharpe ratio calculation to a 60-day period to capture longer-term risk-adjusted performance.
  $$
  \text{Sharpe Ratio}_{60} = \frac{\overline{\text{R}_{60}} - \text{R}_{f}}{\text{VOL60}}
  $$
  where $\overline{\text{R}_{60}}$ is the average return over 60 days, and $\text{VOL60}$ is the standard deviation over 60 days.

### 8. Momentum Factors

- **20-Day Rate of Change (ROC20)**:
  This factor calculates the percentage change in price over the past 20 days, indicating the momentum of the stock.
  $$
  \text{ROC}_{20} = \frac{\text{Close}_{\text{today}} - \text{Close}_{\text{20 days ago}}}{\text{Close}_{\text{20 days ago}}}
  $$

- **One-Month Trading Volume (Volume1M)**:
  This factor sums the trading volume over the past 20 trading days, reflecting the total trading activity in the past month.
  $$
  \text{Volume1M} = \sum_{i=1}^{20} \text{Volume}_{i}
  $$

- **10-Day Triple Exponential Moving Average (TRIX10)**:
  TRIX is a momentum indicator that measures the percentage rate of change of a triple exponentially smoothed moving average. It is calculated using a 10-day period:
  $$
  \text{TRIX}_{10} = \frac{\text{EMA}_{3}(\text{EMA}_{2}(\text{EMA}_{1}(\text{Close})))}{\text{EMA}_{3}(\text{EMA}_{2}(\text{EMA}_{1}(\text{Close}))) - 1}
  $$
  where $\text{EMA}_{1}$, $\text{EMA}_{2}$, and $\text{EMA}_{3}$ are exponentially weighted moving averages with different spans.

- **One-Month Price Change (Price1M)**:
  This factor measures the percentage change in the stock price over the last month (20 trading days).
  $$
  \text{Price1M} = \text{Close}_{\text{today}} - \text{Close}_{\text{20 days ago}}
  $$

- **12-Day Price Momentum (PLRC12)**:
  This factor measures the price momentum by comparing the 12-day moving average with the price 12 days ago:
  $$
  \text{PLRC12} = \frac{\text{MA}_{12}(\text{Close})}{\text{Close}_{\text{12 days ago}}} - 1
  $$

### 9. Technical Factors

- **20-Day MACD (MAC20)**:
  MACD (Moving Average Convergence Divergence) is calculated as the difference between a 12-day and 26-day exponential moving average (EMA) of the close price.
  $$
  \text{MACD}_{20} = \text{EMA}_{12}(\text{Close}) - \text{EMA}_{26}(\text{Close})
  $$

- **Bollinger Bands (boll_up, boll_down)**:
  Bollinger Bands consist of a 20-day simple moving average (SMA) and two bands that are 2 standard deviations away from the SMA.
  $$
  \text{Bollinger Upper Band} = \text{SMA}_{20} + 2 \times \text{VOL20}
  $$
  $$
  \text{Bollinger Lower Band} = \text{SMA}_{20} - 2 \times \text{VOL20}
  $$

- **14-Day Money Flow Index (MFI14)**:
  The MFI14 is a volume-weighted RSI indicator that identifies overbought or oversold conditions.
  1. Calculate the typical price:
  $$
  \text{Typical Price} = \frac{\text{High} + \text{Low} + \text{Close}}{3}
  $$
  2. Calculate raw money flow:
  $$
  \text{Raw Money Flow} = \text{Typical Price} \times \text{Volume}
  $$
  3. Calculate money flow ratio:
  $$
  \text{Money Flow Ratio} = \frac{\sum (\text{Positive Money Flow})}{\sum (\text{Negative Money Flow})}
  $$
  4. Calculate MFI14:
  $$
  \text{MFI14} = 100 - \frac{100}{1 + \text{Money Flow Ratio}}
  $$

After calculating all the factors for each ticker, the data is compiled and saved for further analysis. This enriched dataset can then be used for various machine learning and quantitative analysis applications to evaluate the performance and characteristics of each stock in the S&P 500 index.
"""

#Factor Calculation
print(len(factor_list))

factor_list

# Set pandas option to handle downcasting warning
pd.set_option('future.no_silent_downcasting', True)

#Quality Factors
def calculate_net_profit_to_revenue(stock_data):
    """Calculate net profit to total revenue ratio"""
    if 'netIncome_ttm' in stock_data.columns and 'revenue_ttm' in stock_data.columns:
        return (stock_data['netIncome_ttm'].astype('float64').div(
            stock_data['revenue_ttm'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_roe(stock_data):
    """Calculate Return on Equity"""
    if 'netIncome_ttm' in stock_data.columns and 'totalStockholdersEquity' in stock_data.columns:
        return (stock_data['netIncome_ttm'].astype('float64').div(
            stock_data['totalStockholdersEquity'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_roa(stock_data):
    """Calculate Return on Assets"""
    if 'netIncome_ttm' in stock_data.columns and 'totalAssets' in stock_data.columns:
        stock_data['prev_totalAssets'] = stock_data['totalAssets'].shift(1)
        stock_data['avg_totalAssets'] = ((stock_data['totalAssets'].astype('float64') +
                                        stock_data['prev_totalAssets'].astype('float64')) / 2)
        return (stock_data['netIncome_ttm'].astype('float64').div(
            stock_data['avg_totalAssets'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_gmi(stock_data):
    """Calculate Gross Margin Index"""
    if 'grossProfit_ttm' in stock_data.columns and 'revenue_ttm' in stock_data.columns:
        stock_data['gross_margin'] = (stock_data['grossProfit_ttm'].astype('float64').div(
            stock_data['revenue_ttm'].replace(0, np.nan).astype('float64')
        ))
        stock_data['prev_gross_margin'] = stock_data['gross_margin'].shift(1)
        return (stock_data['gross_margin'] - stock_data['prev_gross_margin']).astype('float64')
    return None

def calculate_acca(stock_data):
    """Calculate Accruals to Assets"""
    if all(col in stock_data.columns for col in ['netIncome_ttm', 'operatingCashFlow_ttm', 'totalAssets']):
        return ((stock_data['netIncome_ttm'].astype('float64') -
                stock_data['operatingCashFlow_ttm'].astype('float64')).div(
            stock_data['totalAssets'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_debt_to_asset_ratio(stock_data):
    """Calculate Debt to Asset Ratio"""
    if 'totalDebt' in stock_data.columns and 'totalAssets' in stock_data.columns:
        return (stock_data['totalDebt'].astype('float64').div(
            stock_data['totalAssets'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

#Value Factors
def calculate_financial_liability(stock_data):
    """Calculate financial liability"""
    if 'totalLiabilities' in stock_data.columns:
        return stock_data['totalLiabilities'].astype('float64')
    return None

def calculate_cash_flow_to_price_ratio(stock_data):
    """Calculate cash flow to price ratio"""
    if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'weightedAverageShsOut', 'close']):
        return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
            stock_data['weightedAverageShsOut'].replace(0, np.nan).astype('float64')
        ).div(stock_data['close'].replace(0, np.nan).astype('float64'))).astype('float64')
    return None

def calculate_price_to_book_ratio(stock_data):
    """Calculate price to book ratio"""
    if all(col in stock_data.columns for col in ['close', 'totalStockholdersEquity', 'weightedAverageShsOut']):
        return (stock_data['close'].astype('float64').div(
            stock_data['totalStockholdersEquity'].astype('float64').div(
                stock_data['weightedAverageShsOut'].replace(0, np.nan).astype('float64')
            )
        )).astype('float64')
    return None

def calculate_price_to_sales_ratio(stock_data):
    """Calculate price to sales ratio"""
    if all(col in stock_data.columns for col in ['close', 'revenue_ttm', 'weightedAverageShsOut']):
        return (stock_data['close'].astype('float64').div(
            stock_data['revenue_ttm'].astype('float64').div(
                stock_data['weightedAverageShsOut'].replace(0, np.nan).astype('float64')
            )
        )).astype('float64')
    return None

def calculate_price_to_earning_ratio(stock_data):
    """Calculate price to earning ratio"""
    if all(col in stock_data.columns for col in ['close', 'eps_ttm']):
        return (stock_data['close'].astype('float64').div(
            stock_data['eps_ttm'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_total_liability_to_total_asset_ratio(stock_data):
    """Calculate total liability to total asset ratio"""
    if all(col in stock_data.columns for col in ['totalLiabilities', 'totalAssets']):
        return (stock_data['totalLiabilities'].astype('float64').div(
            stock_data['totalAssets'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_net_profit(stock_data):
    """Calculate net profit"""
    if 'netIncome_ttm' in stock_data.columns:
        return stock_data['netIncome_ttm'].astype('float64')
    return None

def calculate_working_capital_ratio(stock_data):
    """Calculate working capital ratio"""
    if all(col in stock_data.columns for col in ['totalCurrentAssets', 'totalCurrentLiabilities']):
        return (stock_data['totalCurrentAssets'].astype('float64').div(
            stock_data['totalCurrentLiabilities'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_quick_ratio(stock_data):
    """Calculate quick ratio"""
    if all(col in stock_data.columns for col in ['totalCurrentAssets', 'inventory', 'totalCurrentLiabilities']):
        return ((stock_data['totalCurrentAssets'].astype('float64') -
                stock_data['inventory'].astype('float64')).div(
            stock_data['totalCurrentLiabilities'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_debt_to_equity_ratio(stock_data):
    """Calculate debt to equity ratio"""
    if all(col in stock_data.columns for col in ['totalLiabilities', 'totalStockholdersEquity']):
        return (stock_data['totalLiabilities'].astype('float64').div(
            stock_data['totalStockholdersEquity'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_operate_cash_flow_to_total_asset_ratio(stock_data):
    """Calculate operating cash flow to total asset ratio"""
    if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'totalAssets']):
        return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
            stock_data['totalAssets'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_operate_cash_flow_to_total_liabilities_ratio(stock_data):
    """Calculate operating cash flow to total liabilities ratio"""
    if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'totalLiabilities']):
        return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
            stock_data['totalLiabilities'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_operate_cash_flow_to_net_profit_ratio(stock_data):
    """Calculate operating cash flow to net profit ratio"""
    if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'netIncome_ttm']):
        return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
            stock_data['netIncome_ttm'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_EV_to_operate_cash_flow_ratio(stock_data):
    """Calculate EV to operating cash flow ratio"""
    if all(col in stock_data.columns for col in ['close', 'weightedAverageShsOut', 'totalDebt', 'cashAndCashEquivalents', 'operatingCashFlow_ttm']):
        return ((stock_data['close'].astype('float64') *
                stock_data['weightedAverageShsOut'].astype('float64') +
                stock_data['totalDebt'].astype('float64') -
                stock_data['cashAndCashEquivalents'].astype('float64')).div(
            stock_data['operatingCashFlow_ttm'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

def calculate_debt_to_EBITDA_ratio(stock_data):
    """Calculate debt to EBITDA ratio"""
    if all(col in stock_data.columns for col in ['totalDebt', 'ebitda_ttm']):
        return (stock_data['totalDebt'].astype('float64').div(
            stock_data['ebitda_ttm'].replace(0, np.nan).astype('float64')
        )).astype('float64')
    return None

#Growth Factor
def calculate_eps_growth_rate_ttm(stock_data):
    """Calculate EPS growth rate"""
    if 'eps_ttm' in stock_data.columns:
        return (stock_data['eps_ttm'] / stock_data['eps_ttm'].shift(1) - 1).astype('float64')
    return None

def calculate_peg_ttm(stock_data):
    """Calculate PEG ratio"""
    if all(col in stock_data.columns for col in ['close', 'eps_ttm']):
        # Calculate P/E ratio
        pe_ratio = stock_data['close'] / stock_data['eps_ttm']
        # Calculate EPS growth rate
        eps_ttm_growth = (stock_data['eps_ttm'] / stock_data['eps_ttm'].shift(1) - 1)
        # Calculate PEG ratio
        peg_ratio = pe_ratio / eps_ttm_growth.replace(0, np.nan)
        return peg_ratio.astype('float64')
    return None

def calculate_net_profit_growth_rate_ttm(stock_data):
    """Calculate net profit growth rate"""
    if 'netIncome_ttm' in stock_data.columns:
        return (stock_data['netIncome_ttm'] / stock_data['netIncome_ttm'].shift(1) - 1).astype('float64')
    return None

def calculate_revenue_growth_rate_ttm(stock_data):
    """Calculate revenue growth rate"""
    if 'revenue_ttm' in stock_data.columns:
        return (stock_data['revenue_ttm'] / stock_data['revenue_ttm'].shift(1) - 1).astype('float64')
    return None

def calculate_net_asset_growth_rate(stock_data):
    """Calculate net asset growth rate"""
    if 'totalStockholdersEquity' in stock_data.columns:
        return (stock_data['totalStockholdersEquity'] / stock_data['totalStockholdersEquity'].shift(1) - 1).astype('float64')
    return None

def calculate_operate_cash_flow_growth_rate_ttm(stock_data):
    """Calculate operating cash flow growth rate"""
    if 'operatingCashFlow_ttm' in stock_data.columns:
        return (stock_data['operatingCashFlow_ttm'] / stock_data['operatingCashFlow_ttm'].shift(1) - 1).astype('float64')
    return None

#Stock Factor
def calculate_net_asset_per_share(stock_data):
    """Calculate net asset per share"""
    if all(col in stock_data.columns for col in ['totalStockholdersEquity', 'weightedAverageShsOut']):
        return (stock_data['totalStockholdersEquity'] / stock_data['weightedAverageShsOut']).astype('float64')
    return None

def calculate_net_operate_cash_flow_per_share(stock_data):
    """Calculate net operating cash flow per share"""
    if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'weightedAverageShsOut']):
        return (stock_data['operatingCashFlow_ttm'] / stock_data['weightedAverageShsOut']).astype('float64')
    return None

def calculate_retained_earnings_per_share(stock_data):
    """Calculate retained earnings per share"""
    if all(col in stock_data.columns for col in ['retainedEarnings', 'weightedAverageShsOut']):
        return (stock_data['retainedEarnings'] / stock_data['weightedAverageShsOut']).astype('float64')
    return None

def calculate_market_cap(stock_data):
    """Calculate market capitalization"""
    if all(col in stock_data.columns for col in ['close', 'weightedAverageShsOut']):
        return (stock_data['close'] * stock_data['weightedAverageShsOut']).astype('float64')
    return None

def calculate_liquidity(stock_data):
    """Calculate liquidity"""
    if all(col in stock_data.columns for col in ['volume', 'weightedAverageShsOut']):
        return (stock_data['volume'] / stock_data['weightedAverageShsOut']).astype('float64')
    return None


# Dictionary mapping factor names to their calculation functions
factor_calculations = {
    #Quality Factors
    'net_profit_to_total_revenue_ttm': calculate_net_profit_to_revenue,
    'roe_ttm': calculate_roe,
    'roa_ttm': calculate_roa,
    'GMI': calculate_gmi,
    'ACCA': calculate_acca,
    'debt_to_asset_ratio': calculate_debt_to_asset_ratio,
    #Value Factors
    'financial_liability': calculate_financial_liability,
    'cash_flow_to_price_ratio_ttm': calculate_cash_flow_to_price_ratio,
    'price_to_book_ratio': calculate_price_to_book_ratio,
    'price_to_sales_ratio_ttm': calculate_price_to_sales_ratio,
    'price_to_earning_ratio_ttm': calculate_price_to_earning_ratio,
    'total_liability_to_total_asset_ratio': calculate_total_liability_to_total_asset_ratio,
    'net_profit_ttm': calculate_net_profit,
    'working_capital_ratio': calculate_working_capital_ratio,
    'quick_ratio': calculate_quick_ratio,
    'debt_to_equity_ratio': calculate_debt_to_equity_ratio,
    'operate_cash_flow_to_total_asset_ratio': calculate_operate_cash_flow_to_total_asset_ratio,
    'operate_cash_flow_to_total_liabilities_ratio': calculate_operate_cash_flow_to_total_liabilities_ratio,
    'operate_cash_flow_to_net_profit_ratio': calculate_operate_cash_flow_to_net_profit_ratio,
    'EV_to_operate_cash_flow_ratio': calculate_EV_to_operate_cash_flow_ratio,
    'debt_to_EBITDA_ratio': calculate_debt_to_EBITDA_ratio,
    #Growth Factor
    'EPS_growth_rate_ttm': calculate_eps_growth_rate_ttm,
    'PEG_ttm': calculate_peg_ttm,
    'net_profit_growth_rate_ttm': calculate_net_profit_growth_rate_ttm,
    'revenue_growth_rate_ttm': calculate_revenue_growth_rate_ttm,
    'net_asset_growth_rate': calculate_net_asset_growth_rate,
    'operate_cash_flow_growth_rate_ttm': calculate_operate_cash_flow_growth_rate_ttm,
    #Stock Factor
    'net_asset_per_share': calculate_net_asset_per_share,
    'net_operate_cash_flow_per_share': calculate_net_operate_cash_flow_per_share,
    'retained_earnings_per_share': calculate_retained_earnings_per_share,
    'market_cap(size)': calculate_market_cap,
    #Style Factor
    'liquidity': calculate_liquidity
}

# Create a copy of the DataFrame
financial_factor_df = merged_df_with_ttm_and_price.copy()

# Add new factor columns with NaN values
for col in factor_calculations.keys():
    financial_factor_df[col] = pd.Series(np.nan, index=financial_factor_df.index, dtype='float64')

# Calculate factors for each stock
for symbol in financial_factor_df['symbol'].unique():
    stock_data = financial_factor_df[financial_factor_df['symbol'] == symbol].sort_values('date')

    for factor_name, calc_function in factor_calculations.items():
        try:
            result = calc_function(stock_data)
            if result is not None:
                financial_factor_df.loc[stock_data.index, factor_name] = result
            else:
                print(f"Error: Required columns for {factor_name} not found for {symbol}")
        except Exception as e:
            print(f"Error calculating {factor_name} for {symbol}: {str(e)}")

# Handle any remaining invalid values
for col in factor_calculations.keys():
    financial_factor_df[col] = (
        financial_factor_df[col]
        .replace([np.inf, -np.inf], np.nan)
        .astype('float64')
    )

# Display sample results
print("\nSample results:")

display(financial_factor_df.head())

#Number of Null Values for Each Factor
financial_factor_df[factor_calculations.keys()].isna().sum()

#Stockwise Null Values
factor_list = list(factor_calculations.keys())
stock_wise_nulls = financial_factor_df.groupby('symbol')[factor_list].apply(lambda x: x.isna().sum())
# Add total null values column
stock_wise_nulls['Total_Null_Values'] = stock_wise_nulls.sum(axis=1)
stock_wise_nulls = stock_wise_nulls.sort_values('Total_Null_Values', ascending=False)

display(stock_wise_nulls)

# Print factors for a specific stock and period
factor_df2 = financial_factor_df[['symbol', 'date', 'netIncome_ttm', 'net_profit_ttm', 'operatingCashFlow_ttm'] + list(factor_calculations.keys())]

def print_stock_factors(factor_df2, symbol, start_date=None, end_date=None):
    """
    Print factors for a specific stock and date range

    Parameters:
    - factor_df: DataFrame containing the factors
    - symbol: Stock symbol to filter
    - start_date: Start date for filtering (optional)
    - end_date: End date for filtering (optional)
    """
    # Filter for the specific stock
    stock_data = factor_df2[factor_df2['symbol'] == symbol].copy()

    # Convert date column to datetime if it's not already
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    # Apply date filters if provided
    if start_date:
        stock_data = stock_data[stock_data['date'] >= pd.to_datetime(start_date)]
    if end_date:
        stock_data = stock_data[stock_data['date'] <= pd.to_datetime(end_date)]

    # Sort by date
    stock_data = stock_data.sort_values('date', ascending=False)

    # Select relevant columns
    #display_columns = ['date'] + factor_columns

    print(f"\nFactors for {symbol}:")
    print("=" * 100)
    #print(stock_data[display_columns].to_string())
    print(stock_data.to_string())
    print("=" * 100)

# Example usage:
# Print factors for a specific stock and period
print_stock_factors(factor_df2.round(4), 'DECK', '2023-01-01', '2024-12-31')

#Save the Quarterly Factors
quarterly_financial_factors = financial_factor_df[['symbol', 'date'] + list(factor_calculations.keys())]
display(quarterly_financial_factors.head())

quarterly_financial_factors.to_csv('quarterly_financial_factors.csv', index=False)

print(len(quarterly_financial_factors.columns))
print(len(quarterly_tech_factors.columns))

quarterly_financial_factors.columns

quarterly_dates

# First, create a function to map dates to quarter ends
def map_to_quarter_end(date):
    date = pd.to_datetime(date)
    year = date.year
    month = date.month

    if 1 <= month <= 3:
        return pd.Timestamp(f"{year}-03-31")
    elif 4 <= month <= 6:
        return pd.Timestamp(f"{year}-06-30")
    elif 7 <= month <= 9:
        return pd.Timestamp(f"{year}-09-30")
    else:  # 10 <= month <= 12
        return pd.Timestamp(f"{year}-12-31")

# Create a copy of quarterly_financial_factors and map its dates
financial_factors_mapped = quarterly_financial_factors.copy()
financial_factors_mapped['mapped_date'] = financial_factors_mapped['date'].apply(map_to_quarter_end)

# Create a copy of quarterly_tech_factors
tech_factors_mapped = quarterly_tech_factors.copy()
tech_factors_mapped['mapped_date'] = tech_factors_mapped['date']

# Merge the DataFrames using the mapped dates
quarterly_merged_factors = pd.merge(
    financial_factors_mapped,
    tech_factors_mapped,
    on=['symbol', 'mapped_date'],
    how='inner',
    suffixes=('_financial', '_tech')
)

# Drop the original date columns and rename mapped_date to date
quarterly_merged_factors = quarterly_merged_factors.drop(['date_financial', 'date_tech'], axis=1)
quarterly_merged_factors = quarterly_merged_factors.rename(columns={'mapped_date': 'date'})

# Sort by symbol and date (latest first)
quarterly_merged_factors = quarterly_merged_factors.sort_values(['symbol', 'date'], ascending=[True, False]).reset_index(drop=True)

columns = quarterly_merged_factors.columns.tolist()
columns.remove('date')
columns = ['date'] + columns

# Reorder the columns
quarterly_merged_factors = quarterly_merged_factors[columns]

# Display results
print("\nFirst few rows of merged factors:")
display(quarterly_merged_factors.head())

print("\nAll quarterly merged factors are saved to 'quarterly_merged_factors.csv'")
quarterly_merged_factors.to_csv('quarterly_merged_factors.csv', index=False)

print("Shape of merged factors:", quarterly_merged_factors.shape)
# Show the date range
print("\nDate range in merged factors:")
print("Earliest date:", quarterly_merged_factors['date'].min())
print("Latest date:", quarterly_merged_factors['date'].max())

# Check for any missing values
print("\nMissing values in Quarterly Merged Factors:")
display(quarterly_merged_factors.isna().sum())

quarterly_merged_factors.columns

unique_dates = sorted(quarterly_merged_factors['date'].unique(), reverse=True)
for date in unique_dates:
    print(f"{date.strftime('%Y-%m-%d')} - Quarter: Q{(date.month-1)//3 + 1}")

#Stock wise NaN Values in Quarterly_Marged_factors
def analyze_NaN_values(factor_df):
    factor_columns = [col for col in factor_df.columns if col not in ['date', 'symbol']]

    # Calculate total NaN values per stock
    stock_nan_totals = factor_df.groupby('symbol')[factor_columns].apply(
        lambda x: x.isna().sum().sum()
    ).reset_index()
    stock_nan_totals.columns = ['Symbol', 'Total NaN Values']

    # Sort by total NaN values in descending order
    stock_nan_totals = stock_nan_totals.sort_values('Total NaN Values', ascending=False)

    # Calculate detailed statistics for each factor
    detailed_stats = []
    for symbol in factor_df['symbol'].unique():
        stock_data = factor_df[factor_df['symbol'] == symbol]

        for col in factor_columns:
            nan_count = stock_data[col].isna().sum()
            total_count = len(stock_data)
            percentage = (nan_count / total_count) * 100

            if percentage > 0:  # Only include factors with missing values
                detailed_stats.append({
                    'Symbol': symbol,
                    'Factor Name': col,
                    'Total Values': total_count,
                    'NaN Values': nan_count,
                    'Percentage NaN': f"{percentage:.2f}%"
                })

    return stock_nan_totals, pd.DataFrame(detailed_stats)

def print_stock_wise_stats(factor_df):
    # Get statistics
    total_nan_stats, detailed_stats = analyze_NaN_values(factor_df)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("=" * 100)
    print("\nTotal NaN Values per Stock (Sorted by Missing Values):")
    print(total_nan_stats.to_string(index=False))
    print("\n")

    # Print detailed statistics if available
    if not detailed_stats.empty:
        print("=== DETAILED STATISTICS ===")
        print("=" * 100)
        print("\nDetailed Factor-wise Missing Values:")
        print(detailed_stats.sort_values(['Symbol', 'Factor Name']).to_string(index=False))

    else:
        print("\nNo stocks have missing factor values!")

# Print stock-wise statistics
print_stock_wise_stats(quarterly_merged_factors)

def print_rows_with_nan_factors(factor_df):
    """
    Print rows that have NaN values in any of the factor columns
    """
    # Get all columns except 'date' and 'symbol'
    factor_columns = [col for col in factor_df.columns if col not in ['date', 'symbol']]

    # Find rows with any NaN values in factor columns
    nan_rows = factor_df[factor_df[factor_columns].isna().any(axis=1)]

    if not nan_rows.empty:
        print("\nRows with Missing Factor Values:")
        print("=" * 100)

        # Sort by symbol and date (latest first)
        nan_rows = nan_rows.sort_values(['symbol', 'date'], ascending=[True, False])

        # Display the rows
        print(nan_rows.to_string())
        print("=" * 100)

        # Print summary of how many rows have NaN values
        print(f"\nTotal number of rows with missing factors: {len(nan_rows)}")

        # Save to CSV
        nan_rows_csv_path = 'rows_with_missing_factors.csv'
        nan_rows.to_csv(nan_rows_csv_path, index=False)
        print(f"\nRows with missing factors saved to: {nan_rows_csv_path}")
    else:
        print("\nNo rows have missing factor values!")

# Print rows with NaN factors
print_rows_with_nan_factors(quarterly_merged_factors)

# Calculate Quarterly Returns
combined_df_return_4 = combined_df.copy()

# Ensure date is datetime
combined_df_return_4['date'] = pd.to_datetime(combined_df_return_4['date'])

# Set date and symbol as index
combined_df_return_4.set_index(['date', 'symbol'], inplace=True)

# Get close prices and unstack to have symbols as columns
close_prices = combined_df_return_4['close'].unstack(level=-1)

# Resample to quarterly frequency and get last price of each quarter
quarterly_prices = close_prices.resample('QE').last()  # Get last price of each quarter

# Calculate percentage change between current quarter and previous quarter
quarterly_returns = quarterly_prices.pct_change().dropna(axis=0, how='all')  # This calculates (current - previous) / previous
quarterly_returns['factor'] = 'quarterly_return'

# Reset index to get date as a column and Melt the dataframe to get it in long format
quarterly_returns = quarterly_returns.reset_index().melt(
    id_vars=['date', 'factor'],
    var_name='symbol',
    value_name='Quarterly_Return'
)

# Pivot the reshaped data back to have 'date' and 'factor' as index and symbols as columns
quarterly_returns = quarterly_returns.pivot(
    index=['date', 'factor'],
    columns='symbol',
    values='Quarterly_Return'
)

# Sort by date in descending order (most recent first)
quarterly_returns = quarterly_returns.sort_index(level='date', ascending=False)

# Display the first few rows of quarterly returns
print("Quarterly Returns:")
display(quarterly_returns.head())

# Save to CSV
quarterly_returns.to_csv('quarterly_returns.csv')

# Calculate percentage change and shift forward to get next quarter's returns
next_quarter_returns = quarterly_prices.pct_change().dropna(axis=0, how='all')
next_quarter_returns = next_quarter_returns.shift(1)  # Shift forward to get next quarter's returns
next_quarter_returns.dropna(axis=0, how='all', inplace=True)
next_quarter_returns['factor'] = 'next_quarter_return'

# Reset index to get date as a column and Melt the dataframe to get it in long format
next_quarter_returns = next_quarter_returns.reset_index().melt(
    id_vars=['date', 'factor'],
    var_name='symbol',
    value_name='next_quarter_return'
)

# Pivot the reshaped data back to have 'date' and 'factor' as index and symbols as columns
next_quarter_returns = next_quarter_returns.pivot(
    index=['date', 'factor'],
    columns='symbol',
    values='next_quarter_return'
)

# Sort by date in descending order (most recent first)
next_quarter_returns = next_quarter_returns.sort_index(level='date', ascending=False)

# Display the first few rows of next quarter returns
print("Next Quarter's Returns:")
display(next_quarter_returns.head())

# Save to CSV
next_quarter_returns.to_csv('next_quarter_returns.csv')

# Convert quarterly_merged_factors to pivot format
quarterly_merged_factors = quarterly_merged_factors.reset_index()
quarterly_merged_factors = quarterly_merged_factors.melt(
    id_vars=['date', 'symbol'],
    var_name='factor',
    value_name='value'
)
quarterly_merged_factors = quarterly_merged_factors.pivot_table(
    index=['date', 'factor'],
    columns='symbol',
    values='value'
)

# 2. Merge all three datasets
combined_quarterly_data = pd.concat([
    quarterly_merged_factors,  # Factor data
    quarterly_returns,         # Current quarter returns
    next_quarter_returns       # Next quarter returns
])

# Sort the data by date (most recent first) and factor
combined_quarterly_data = combined_quarterly_data.sort_index(level=['date', 'factor'], ascending=[False, True])

# Save to CSV
#combined_quarterly_data.to_csv('combined_quarterly_data.csv', index=False)

# Check shape and columns of combined_data
print(f"Shape of combined_data: {combined_quarterly_data.shape}")
print(f"Columns in combined_data: {combined_quarterly_data.columns.tolist()}")
print(f"Start Date: {combined_quarterly_data.index.get_level_values('date').max()}")
print(f"End Date: {combined_quarterly_data.index.get_level_values('date').min()}")
print(f"Index levels in combined_data: {combined_quarterly_data.index.names}")

# Display the first few rows of combined data
print("Combined Quarterly Data:")
display(combined_quarterly_data.head().round(4))

# Drop the 'index' factor from the DataFrame
combined_quarterly_data = combined_quarterly_data[combined_quarterly_data.index.get_level_values(1) != 'index']

# Print factors to verify the change
print("\nFactors in the Dataset:")
print("=" * 50)

# Get unique factors from the index
factors = combined_quarterly_data.index.get_level_values(1).unique()

# Print each factor on a new line with numbering
for i, factor in enumerate(factors, 1):
    print(f"{i}. {factor}")

print("=" * 50)
print(f"Total number of factors: {len(factors)}")

# Display the first few rows to verify the format
print("First few rows of Combined Quartely Data:")
display(combined_quarterly_data.head().round(4))

# Save the result to a CSV file
print(f"Combined Quarterly Data saved to {'combined_quarterly_data.csv'}")
combined_quarterly_data.to_csv('combined_quarterly_data.csv')



