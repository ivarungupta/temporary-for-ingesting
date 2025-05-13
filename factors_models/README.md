# Factors_Models Folder Readme

This folder contains the core modules responsible for calculating various financial factors, such as quality, value, risk, and stock-related metrics. Each file corresponds to a specific class that calculates different financial or market-related factors.

---

### **1. `factors.py`**

#### **Files Where This File Is Being Called:**
- `main.py` (for factor calculations)

#### **Files This File Is Calling:**
- `FMPWrapper` from `data_sources/fmp.py`
- `Quality`, `Value`, `Stock`, `Growth`, `Emotional`, `Style`, `Risk`, `Momentum`, `Technical` classes from within the same `models` folder.

#### **API Usage:**
- Uses `FMPWrapper` to fetch financial data, such as income statements, balance sheets, cash flow data, and financial ratios.
  
#### **Input to the File:**
- `ticker` (Stock ticker symbol)
- `fmp` (Instance of `FMPWrapper`)
- `start_date`, `end_date` (Time range for fetching data)

#### **Output / What the File Returns:**
- Returns a dictionary containing calculated factors for each financial category (e.g., `quality`, `value`, `growth`, etc.).

#### **What Exactly This File Is Performing and How It Is Performing:**
- **FactorsWrapper Class**: This class aggregates and calculates all factors for a given stock ticker using data from the `FMPWrapper`.
  - It fetches the financial data and market data for the stock.
  - For each factor type (e.g., Quality, Value, Growth, etc.), it instantiates the corresponding class and calls their `calculate_all_factors()` method.
  - The results are stored in a dictionary, which is returned as the final output, containing the calculated financial factors for the stock over the given time period.

---

### **2. `growth.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating growth-related factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `income_data` (Income statement data)
- `balance_data` (Balance sheet data)
- `cash_flow_data` (Cash flow data)
- `market_data` (Market data, including stock prices)

#### **Output / What the File Returns:**
- Returns a DataFrame with growth factors such as PEG ratio, revenue growth, net profit growth, net asset growth, operating cash flow growth, and EPS growth rate.

#### **What Exactly This File Is Performing and How It Is Performing:**

- **Growth Class**: Calculates financial growth metrics based on income, balance, and cash flow data.

  - **Methods**:

    - `__init__(self, income_data, balance_data, cash_flow_data, market_data)`: 
      - Initializes the class with the provided financial data (`income_data`, `balance_data`, `cash_flow_data`, `market_data`).
      - Validates the columns of the input data using the `_validate_columns()` method.
    
    - `_validate_columns(self)`: 
      - Ensures that all required columns (`'eps'`, `'netIncome'`, `'revenue'`, `'totalStockholderEquity'`, etc.) are present in the input data. Raises a `ValueError` if any column is missing.

    - `safe_get_value(self, df, column)`: 
      - Safely retrieves the first value of the specified column in the dataframe, or returns `np.nan` if the value is missing or the column doesn't exist.
      
    - `safe_get_value_ttm(self, df, column)`: 
      - Retrieves the sum of the last four available quarters (Trailing Twelve Months) for the specified column. Returns `np.nan` if there aren’t enough data points.

    - `safe_get_value_ttm_3(self, df, column)`: 
      - Retrieves the sum of the last three available quarters (Trailing Twelve Months) for the specified column.

    - `calculate_peg(self)`: 
      - **Purpose**: Calculates the Price-to-Earnings Growth (PEG) ratio.
      - **How**: 
        - Uses `market_data` for the stock price and `income_data` for EPS.
        - Computes the PEG ratio by dividing the Price/Earnings ratio by the EPS growth rate.
    
    - `calculate_net_profit_growth(self)`: 
      - **Purpose**: Calculates the net profit growth.
      - **How**: 
        - Uses `income_data` to compute the percentage change in net profit between consecutive periods.
    
    - `calculate_revenue_growth(self)`: 
      - **Purpose**: Calculates the revenue growth.
      - **How**: 
        - Uses `income_data` to compute the percentage change in revenue between consecutive periods.
    
    - `calculate_net_asset_growth(self)`: 
      - **Purpose**: Calculates the growth rate of net assets (stockholder equity).
      - **How**: 
        - Uses `balance_data` to compute the percentage change in total stockholder equity between consecutive periods.
    
    - `calculate_operating_cashflow_growth(self)`: 
      - **Purpose**: Calculates the operating cash flow growth.
      - **How**: 
        - Uses `cash_flow_data` to compute the percentage change in operating cash flow between consecutive periods.
    
    - `calculate_eps_growth_rate(self)`: 
      - **Purpose**: Calculates the EPS growth rate.
      - **How**: 
        - Uses `income_data` to calculate the EPS growth rate over the trailing twelve months (TTM).
    
    - `calculate_all_factors(self)`: 
      - **Purpose**: Loops through all the available dates in the `income_data_master`, `balance_data_master`, `cash_flow_data_master`, and `market_data_master` dataframes and calculates all growth factors for each period.
      - **How**: 
        - Iterates through each date in the data, ensures the required previous periods of data exist, and then calculates the PEG ratio, net profit growth, revenue growth, net asset growth, operating cash flow growth, and EPS growth rate.
        - Returns a DataFrame containing all the growth metrics for each period.

---

### **3. `momentum.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating momentum-related factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `df` (DataFrame containing OHLCV market data)

#### **Output / What the File Returns:**
- Returns a DataFrame with momentum-related factors such as rate of change, volume, price change, and price level ratio.

#### **What Exactly This File Is Performing and How It Is Performing:**
- **Momentum Class**: Calculates momentum-related indicators for stock price data.

  - **Methods**:

    - `__init__(self, df)`:
      - **Purpose**: Initializes the class with the given DataFrame (`df`) and validates that it contains the necessary columns (`'close'`, `'volume'`).
      - **Validation**: The `_validate_columns()` method checks if the required columns are present in the DataFrame. If any are missing, it raises a `ValueError`.

    - `_validate_columns(self)`:
      - **Purpose**: Ensures that the DataFrame contains the required columns: `'close'` (closing price) and `'volume'` (volume).
      - **Error Handling**: Raises a `ValueError` if any of the required columns are missing.

    - `calculate_rate_of_change(self, window=60)`:
      - **Purpose**: Computes the Rate of Change (ROC) over a specified window (default is 60).
      - **How**: 
        - The formula for ROC is `(current_close - previous_close) / previous_close`.
        - It shifts the closing prices by the given window size and calculates the percentage change.
      - **Output**: Adds a new column `'ROC60'` to the DataFrame representing the rate of change.

    - `calculate_volume_quarterly(self, window=60)`:
      - **Purpose**: Sums the trading volume over a rolling window (default is 60).
      - **How**: 
        - Uses a rolling sum over the given window to calculate the total volume over that period.
      - **Output**: Adds a new column `'Volume1Q'` to the DataFrame representing the volume over the rolling window.

    - `calculate_trix(self, span=30)`:
      - **Purpose**: Computes the TRIX (Triple Exponential Moving Average) indicator.
      - **How**: 
        - The TRIX indicator is calculated by applying an Exponential Moving Average (EMA) three times.
        - The result is the percentage change of the triple EMA.
      - **Output**: Adds a new column `'TRIX30'` to the DataFrame representing the TRIX value.

    - `calculate_price_quarterly(self, window=60)`:
      - **Purpose**: Calculates the price change over a specified window (default is 60).
      - **How**: 
        - The formula is `(current_close - previous_close)`.
        - The closing prices are shifted by the given window and the difference is calculated.
      - **Output**: Adds a new column `'Price1Q'` to the DataFrame representing the price change over the rolling window.

    - `calculate_price_level_ratio(self, window=36)`:
      - **Purpose**: Computes the price level ratio.
      - **How**: 
        - The formula is `moving_average(close) / previous_close - 1`.
        - The method calculates the ratio of the moving average of the closing prices over the given window relative to the previous closing price.
      - **Output**: Adds a new column `'PLRC36'` to the DataFrame representing the price level ratio.

    - `calculate_all_factors(self)`:
      - **Purpose**: Calculates all the momentum factors (ROC, volume, TRIX, price change, and price level ratio).
      - **How**: 
        - Calls the individual methods (`calculate_rate_of_change()`, `calculate_volume_quarterly()`, `calculate_trix()`, `calculate_price_quarterly()`, and `calculate_price_level_ratio()`).
      - **Output**: Returns a DataFrame with the calculated momentum factors (`'ROC60'`, `'Volume1Q'`, `'TRIX30'`, `'Price1Q'`, `'PLRC36'`) along with the date.

---

### **4. `quality.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating quality-related financial factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `income_data`, `balance_data`, `cash_flow_data` (Financial data)

#### **Output / What the File Returns:**
- Returns a DataFrame with quality factors like return on equity (ROE), gross margin increment (GMI), and accruals.

#### **What Exactly This File Is Performing and How It Is Performing:**
- **Quality Class**: Focuses on financial quality metrics such as profitability and financial stability.

  - **Methods**:
    - `calculate_net_profit_to_revenue()`: 
      - **Purpose**: Computes the ratio of net profit to revenue.
      - **How**: Uses `income_data` to calculate the net profit divided by total revenue, which indicates the efficiency of a company in converting revenue to profit.
    
    - `calculate_decm()`:
      - **Purpose**: Measures the difference between current and previous quarter's inventory and account payables as a percentage of total assets.
      - **How**: Uses `balance_data` to calculate the change in inventory and payables, relative to total assets.

    - `calculate_roe()`:
      - **Purpose**: Calculates Return on Equity (ROE).
      - **How**: The formula is `net income / stockholder equity`. Uses `income_data` for net income and `balance_data` for stockholder equity to calculate the ROE.
    
    - `calculate_roa()`:
      - **Purpose**: Calculates Return on Assets (ROA).
      - **How**: The formula is `net income / total assets`. Uses `income_data` for net income and `balance_data` for total assets to calculate the ROA.
    
    - `calculate_gmi()`:
      - **Purpose**: Measures the change in gross margin compared to the previous quarter.
      - **How**: Uses `income_data` to calculate the gross margin for the current and previous periods, then computes the percentage change.
    
    - `calculate_acca()`:
      - **Purpose**: Measures the difference between net income and operating cash flow as a percentage of total assets.
      - **How**: The formula is `(net income - operating cash flow) / total assets`. It uses `income_data` for net income, `cash_flow_data` for operating cash flow, and `balance_data` for total assets.
    
    - `calculate_debtToAsset()`:
      - **Purpose**: Computes the ratio of total debt to total assets.
      - **How**: Uses `balance_data` to calculate the total debt divided by total assets, providing a measure of financial leverage.
    
    - `calculate_all_factors()`:
      - **Purpose**: Iterates over each row of the financial data (income, balance, and cash flow) and calculates the aforementioned quality factors for each date.
      - **How**: For each date in the data, it calls the individual methods (`calculate_net_profit_to_revenue()`, `calculate_roe()`, `calculate_roa()`, etc.) and appends the results to the DataFrame.

  - **Helper Methods**:
    - `_validate_columns()`:
      - **Purpose**: Ensures that all required columns are present in the input data.
      - **How**: Checks for the presence of columns such as `eps`, `netIncome`, `revenue`, `totalStockholderEquity`, `totalAssets`, etc., in the `income_data`, `balance_data`, and `cash_flow_data`. Raises an error if any are missing.
    
    - `safe_get_value_ttm()`:
      - **Purpose**: Safely retrieves the TTM (Trailing Twelve Months) value for specific columns.
      - **How**: Calculates the sum of the last four quarters for the specified column in `income_data`, `balance_data`, or `cash_flow_data`.

  - The class processes the provided financial data, calculates various quality metrics for each quarter, and returns them as a DataFrame.
  
---

### **5. `risk.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating risk-related factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `df` (DataFrame containing market data, specifically the closing prices)
- `risk_free_rate_20`, `risk_free_rate_60` (Risk-free rates)

#### **Output / What the File Returns:**
- Returns a DataFrame with risk factors like variance, Sharpe ratios, and kurtosis.

#### **What Exactly This File Is Performing and How It Is Performing:**
- **Risk Class**: Calculates risk metrics to analyze the volatility and stability of the stock.
  - **Methods**:
    - `__init__(df, risk_free_rate_20=0.02, risk_free_rate_60=0.02)`:
      - Initializes the `Risk` class with market data and risk-free rates for 20-day and 60-day windows. It also validates the presence of required columns in the DataFrame.
      
    - `_validate_columns()`:
      - **Purpose**: Ensures that the DataFrame contains the required columns (`'close'` for closing prices).
      - **How**: Checks if the `'close'` column is present, and raises an error if it's missing.
      
    - `calculate_variance(window=60)`:
      - **Purpose**: Computes the variance of price changes over the specified rolling window (default: 60 days).
      - **How**: Uses the percentage change of closing prices, computes the rolling variance over the specified window, and adds it as a new column (`Variance60`).
      
    - `calculate_sharpe_ratio_20()`:
      - **Purpose**: Calculates the Sharpe ratio using a 20-day window.
      - **How**: The Sharpe ratio is calculated as `(mean return - risk-free rate) / standard deviation of return`. The 20-day rolling window is used for the calculation.
      
    - `calculate_kurtosis(window=60)`:
      - **Purpose**: Measures the kurtosis (tailedness) of the price distribution over the specified window (default: 60 days).
      - **How**: Computes the rolling kurtosis of the percentage change in closing prices, and adds it as a new column (`Kurtosis60`).
      
    - `calculate_skewness(window=60)`:
      - **Purpose**: Computes the skewness (asymmetry) of the price distribution over the specified window (default: 60 days).
      - **How**: Uses the rolling skewness of the percentage change in closing prices and adds it as a new column (`Skewness60`).
      
    - `calculate_sharpe_ratio_60()`:
      - **Purpose**: Calculates the Sharpe ratio using a 60-day window.
      - **How**: Similar to `calculate_sharpe_ratio_20()`, but using a 60-day rolling window for returns.

    - `calculate_all_factors()`:
      - **Purpose**: Computes all the risk factors (`Variance60`, `sharpe_ratio_20`, `Kurtosis60`, `Skewness60`, and `sharpe_ratio_60`).
      - **How**: Calls each of the individual factor calculation methods, collects the computed risk factors, and returns a DataFrame containing those factors.

  - **Helper Methods**:
    - **Risk Metrics Calculated**:
      - `Variance60`: Measures the variance of the stock's price changes over a rolling window.
      - `Sharpe Ratio (20-day)`: Indicates the risk-adjusted return over a 20-day window.
      - `Kurtosis60`: Measures the kurtosis (tailedness) of the price distribution over a 60-day window.
      - `Skewness60`: Measures the skewness (asymmetry) of the price distribution over a 60-day window.
      - `Sharpe Ratio (60-day)`: Indicates the risk-adjusted return over a 60-day window.
  
  - The `Risk` class processes the market data and computes various risk metrics for stock price movements, providing insights into volatility, stability, and risk-adjusted returns.


---
### **6. `stock.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating stock-related financial factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `income_data`, `balance_data`, `cash_flow_data`, `market_data` (Financial and market data)

#### **Output / What the File Returns:**
- Returns a DataFrame with stock-related factors like EPS, market cap, and liquidity.

#### **What Exactly This File Is Performing and How It Is Performing:**
- **Stock Class**: Calculates stock-related metrics that help evaluate a stock's financial health.
  - **Methods**:
    - `__init__(self, income_data, balance_data, cash_flow_data, market_data)`:
      - **Purpose**: Initializes the class with financial and market data and validates required columns.
      - **How**: Stores the input data and runs the validation method to ensure required columns exist.
      
    - `_validate_columns(self)`:
      - **Purpose**: Ensures the provided data contains the required columns (`'income_data'`, `'balance_data'`, `'cash_flow_data'`, and `'market_data'`).
      - **How**: Raises an exception if any required columns are missing.
      
    - `safe_get_value(self, df, column)`:
      - **Purpose**: Retrieves a value safely from a DataFrame.
      - **How**: Returns the value from the specified column if available, else returns `NaN`.
      
    - `safe_get_value_ttm(self, df, column)`:
      - **Purpose**: Retrieves the trailing twelve-month (TTM) value for a given column.
      - **How**: Sums the values from the first 4 rows of the specified column and returns the sum.
      
    - `calculate_net_asset_per_share(self)`:
      - **Purpose**: Computes the net assets per share.
      - **How**: Divides the total stockholders' equity by the weighted average shares outstanding and returns the result.
      
    - `calculate_net_operate_cash_flow_per_share(self)`:
      - **Purpose**: Calculates net operating cash flow per share.
      - **How**: Divides the operating cash flow by the weighted average shares outstanding and returns the result.
      
    - `calculate_eps(self)`:
      - **Purpose**: Calculates Earnings Per Share (EPS).
      - **How**: Uses the most recent data to calculate EPS.
      
    - `calculate_retained_earnings_per_share(self)`:
      - **Purpose**: Computes retained earnings per share.
      - **How**: Divides the retained earnings by the weighted average shares outstanding and returns the result.
      
    - `calculate_liquidity(self)`:
      - **Purpose**: Calculates liquidity.
      - **How**: Divides the trading volume by the weighted average shares outstanding and returns the liquidity.
      
    - `calculate_market_cap(self)`:
      - **Purpose**: Computes the market capitalization.
      - **How**: Multiplies the weighted average shares outstanding by the stock's closing price and returns the market cap.
      
    - `calculate_all_factors(self)`:
      - **Purpose**: Calculates all stock-related financial factors.
      - **How**: Loops through each date, calculates all relevant metrics (EPS, market cap, liquidity), and returns the results in a DataFrame.
  
  - **Helper Methods**:
    - **Stock Metrics Calculated**:
      - `Net Asset Per Share`: Total stockholders' equity divided by the weighted average shares outstanding.
      - `Net Operating Cash Flow Per Share`: Operating cash flow divided by the weighted average shares outstanding.
      - `EPS`: Earnings per Share.
      - `Retained Earnings Per Share`: Retained earnings divided by the weighted average shares outstanding.
      - `Liquidity`: Trading volume divided by the weighted average shares outstanding.
      - `Market Cap`: Market capitalization calculated by multiplying the weighted average shares outstanding by the stock's closing price.
  
  - The `Stock` class processes the input data and computes a variety of metrics to assess a company's financial health, including stock-related factors such as EPS, liquidity, and market capitalization. The results are returned in a structured DataFrame for further analysis.

---

### **7. `style.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating style-related market factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `df` (DataFrame containing market data, specifically the stock's price and volume)
- `sp500_returns` (DataFrame containing the S&P 500 returns data)
- `tickers` (List of stock tickers)

#### **Output / What the File Returns:**
- Returns a DataFrame with style factors such as `beta`, `growth`, and `momentum`.

#### **What Exactly This File Is Performing and How It Is Performing:**
- **Style Class**: This class is designed to calculate market style factors, specifically the beta, growth, and momentum of a stock. Here's a breakdown of the methods in the file:
  
  - **`__init__(self, df, sp500_returns, tickers)`**: 
    - **Purpose**: Initializes the class by accepting stock data (`df`), S&P 500 returns data (`sp500_returns`), and stock tickers (`tickers`) for processing.
    - **How**: The constructor stores the provided data into instance variables and calls the `_validate_columns` method to ensure that necessary columns are present.
  
  - **`_validate_columns(self)`**:
    - **Purpose**: Ensures that the required columns (`'close'`, `'volume'`, `'date'`) are present in the provided stock data.
    - **How**: This method checks for the existence of the essential columns needed for calculating the style factors. If any columns are missing, it raises an error.
  
  - **`calculate_beta(self, window_size=62)`**:
    - **Purpose**: Computes the beta coefficient for a stock, which measures its correlation with the S&P 500. Beta reflects how much the stock moves in relation to the broader market.
    - **How**: The method uses a rolling window (default size is 62 days) to calculate the covariance between the stock's returns and the S&P 500 returns, then normalizes it by the market's variance. The result represents how sensitive the stock's price is to the movements of the market.
  
  - **`calculate_growth(self, window=252)`**:
    - **Purpose**: Calculates the cumulative log return over a specific window, typically representing the stock's growth over the period.
    - **How**: It computes the difference between the log-transformed current closing price and the closing price from a previous time window (`log(close) - log(close.shift(window))`). This gives the percentage change in price over the window, providing insight into the stock's growth.
  
  - **`calculate_momentum(self, window=252)`**:
    - **Purpose**: Measures the momentum of the stock's price over a specified window, indicating whether the stock is trending upward or downward.
    - **How**: This method calculates the percentage change in the stock price from a prior period (`(close / close.shift(window)) - 1`). It helps determine if the stock is gaining or losing momentum over time.

  - **`calculate_all_factors(self)`**:
    - **Purpose**: Calls all the methods (`calculate_beta`, `calculate_growth`, and `calculate_momentum`) to compute the style-related factors in one step and returns the results in a DataFrame.
    - **How**: This method sequentially calculates the `beta`, `growth`, and `momentum` values for the stock and returns them together in a consolidated DataFrame, ready for further analysis.

---

### **8. `technical.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating technical factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `df` (DataFrame containing OHLCV market data)

#### **Output / What the File Returns:**
- Returns a DataFrame with technical factors such as MACD, Bollinger Bands, and MFI.

#### **What Exactly This File Is Performing and How It Is Performing:**
- **Technical Class**: Calculates common technical indicators for stock price data.
  - **Methods**:
    - `calculate_mac(fast_span=36, slow_span=78)`: Computes the MACD (Moving Average Convergence Divergence) using exponential moving averages (EMA) with a fast and slow span.
      - *Purpose*: MACD is used to identify changes in the strength, direction, momentum, and duration of a trend in a stock's price.
      - *How*: It calculates the difference between two EMAs (fast and slow).

    - `calculate_bollinger_bands(window=60, num_std=2)`: Calculates Bollinger Bands, which include the upper and lower bands based on a rolling mean and standard deviation.
      - *Purpose*: Bollinger Bands help measure the volatility of a stock. The bands expand during periods of high volatility and contract during low volatility.
      - *How*: It computes the rolling mean and standard deviation, then uses them to calculate the upper and lower bands.

    - `calculate_mfi(window=42)`: Computes the Money Flow Index (MFI), which is an oscillator that uses both price and volume data to measure the flow of money into and out of a stock.
      - *Purpose*: MFI is used to spot potential overbought or oversold conditions based on both price and volume.
      - *How*: It computes the typical price, multiplies it by volume, and compares positive and negative flows to compute the MFI.

    - `calculate_all_factors()`: A method that calls all the individual technical factor calculation methods (`calculate_mac()`, `calculate_bollinger_bands()`, and `calculate_mfi()`), then returns a DataFrame with the selected technical columns.
      - *Purpose*: To compute and return all key technical factors in a single step.
      - *How*: It calculates each of the factors and then selects the relevant columns (`MAC60`, `boll_up`, `boll_down`, and `MFI42`) for the output.

---

### **9. `value.py`**

#### **Files Where This File Is Being Called:**
- `factors.py` (for calculating value-related financial factors)

#### **Files This File Is Calling:**
- No external files are called by this file directly.

#### **API Usage:**
- None.

#### **Input to the File:**
- `income_data`, `balance_data`, `cash_flow_data`, `market_data`, `financial_ratio_data` (Financial and market data)

#### **Output / What the File Returns:**
- Returns a DataFrame with value factors like P/E ratio, market cap, and net profit.

#### **What Exactly This File Is Performing and How It Is Performing:**
- **Value Class**: Calculates value-related financial metrics to assess a stock's intrinsic value.
  - **Methods**:

    1. **`safe_get_value()`**
       - **Purpose**: Safely retrieves a value from a given column in a DataFrame, handling missing or NaN values.
       - **How**: This method accepts a DataFrame and column name as input, checks if the value exists, and returns it. If the value is missing or NaN, it returns zero.

    2. **`safe_get_value_ttm()`**
       - **Purpose**: Retrieves a value from the trailing twelve months (TTM) of a specified column, handling missing data and ensuring the proper time-based calculation.
       - **How**: It looks for the TTM value in the provided DataFrame and column, returning zero if the value is missing or NaN. This ensures the proper treatment of time-series data.

    3. **`calculate_financial_liability()`**
       - **Purpose**: Computes the total financial liabilities of the company, which helps assess its financial stability.
       - **How**: It uses the `safe_get_value()` method to extract the values for total debt, short-term debt, and other liabilities, then calculates the total liability as the sum of these values.

    4. **`calculate_cashflow_price()`**
       - **Purpose**: Computes the operating cash flow to price ratio, which helps assess the company’s ability to generate cash relative to its market price.
       - **How**: This method divides the operating cash flow (retrieved using `safe_get_value_ttm()`) by the stock price to compute the ratio.

    5. **`calculate_priceToBook()`**
       - **Purpose**: Computes the price-to-book (P/B) ratio, a common measure used to assess the market's valuation of a company's equity relative to its book value.
       - **How**: This method divides the stock price by the book value per share (calculated using `safe_get_value_ttm()` for total equity) to compute the P/B ratio.

    6. **`calculate_price_to_sales()`**
       - **Purpose**: Computes the price-to-sales (P/S) ratio, which helps assess the market's valuation of a company's revenue.
       - **How**: This method divides the stock price by the total revenue per share, providing a valuation metric based on revenue.

    7. **`calculate_price_to_earnings()`**
       - **Purpose**: Calculates the price-to-earnings (P/E) ratio, a widely used indicator of stock valuation relative to its earnings.
       - **How**: It divides the stock price by the earnings per share (EPS) from the income data, producing the P/E ratio.

    8. **`calculate_ltd_to_ta()`**
       - **Purpose**: Computes the long-term debt to total assets ratio, which measures a company's financial leverage and risk.
       - **How**: It calculates the ratio by dividing long-term debt by the total assets of the company, providing an indication of financial risk.

    9. **`calculate_net_profit()`**
       - **Purpose**: Calculates the net profit over the trailing twelve months (TTM), which is a key indicator of profitability.
       - **How**: It extracts the net profit data from the income statement using `safe_get_value_ttm()` and returns the calculated value.

    10. **`calculate_ebit()`**
        - **Purpose**: Computes Earnings Before Interest and Taxes (EBIT), a key measure of profitability.
        - **How**: This method calculates EBIT by extracting the necessary data from the income statement using `safe_get_value_ttm()`.

    11. **`calculate_working_capital_ratio()`**
        - **Purpose**: Calculates the working capital ratio, an indicator of a company’s operational efficiency and short-term financial health.
        - **How**: It calculates working capital as the difference between current assets and current liabilities and divides this by total assets.

    12. **`calculate_quick_ratio()`**
        - **Purpose**: Computes the quick ratio, which is used to assess a company's short-term liquidity by excluding inventory from current assets.
        - **How**: This method divides the difference between current assets and inventory by current liabilities to compute the quick ratio.

    13. **`calculate_operating_cashflow_to_total_assets()`**
        - **Purpose**: Measures the efficiency of a company in generating operating cash flow relative to its total assets.
        - **How**: It divides operating cash flow by the total assets (both retrieved using `safe_get_value_ttm()`), providing an efficiency metric.

    14. **`calculate_ev_to_operating_cashflow()`**
        - **Purpose**: Calculates the Enterprise Value to Operating Cash Flow ratio, a measure of the company’s valuation relative to its ability to generate cash.
        - **How**: The method divides the enterprise value by operating cash flow to compute this ratio.

    15. **`calculate_operating_cashflow_to_net_profit()`**
        - **Purpose**: Computes the operating cash flow to net profit ratio, providing insight into the quality of earnings.
        - **How**: This method divides operating cash flow by net profit, indicating how much of the net profit is actually being generated as cash.

    16. **`calculate_debt_to_ebitda()`**
        - **Purpose**: Calculates the debt-to-EBITDA ratio, which helps assess the company’s ability to repay its debt using earnings before interest, taxes, depreciation, and amortization.
        - **How**: This method divides the company’s total debt by EBITDA, giving an indicator of financial leverage and ability to service debt.

    17. **`calculate_debt_to_equity()`**
        - **Purpose**: Measures the company’s financial leverage by comparing its total debt to shareholders' equity.
        - **How**: It divides the total debt by equity, producing a ratio that shows the relative proportion of debt used to finance the company.

    18. **`calculate_all_factors()`**
        - **Purpose**: This method aggregates all the individual calculations into a complete set of value-related metrics for the stock.
        - **How**: It loops through each date in the financial data, applies all the value-related calculation methods, and returns a DataFrame with all the computed factors.

---
