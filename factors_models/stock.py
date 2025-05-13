import pandas as pd
import numpy as np

class Stock:
    """
    A class to calculate stock-related financial factors.
    """
    def __init__(self, income_data, balance_data, cash_flow_data, market_data):
        self.income_data_master = income_data
        self.balance_data_master = balance_data
        self.cash_flow_data_master = cash_flow_data
        self.market_data_master = market_data
        self.required_columns = {
            'income': {'eps', 'weightedAverageShsOut'},
            'balance': {'totalStockholdersEquity', 'retainedEarnings'},
            'cash_flow': {'operatingCashFlow', 'freeCashFlow'},
            'market': {'close'},
        }
        self._validate_columns()

    def _validate_columns(self):
        missing_cols = []
        for df_name, columns in self.required_columns.items():
            df = getattr(self, f"{df_name}_data_master")
            for col in columns:
                if col not in df.columns:
                    missing_cols.append(f"{df_name}: {col}")
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    def safe_get_value(self, df, column):
        if df.empty or column not in df.columns:
            return np.nan
        try:
            return df[column].iloc[0]
        except Exception:
            return np.nan
    
    def safe_get_value_ttm(self, df, column):
        if df.empty or column not in df.columns:
            return np.nan
        try:
            # Ensure there are at least 4 quarters available
            if len(df) < 4:
                return df[column].iloc[:len(df)].sum()
            # Sum the values of the first 4 rows (most recent four quarters)
            return df[column].iloc[:4].sum()
        
        except Exception:
            return np.nan

    def calculate_net_asset_per_share(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        return self.safe_get_value(self.balance_data, 'totalStockholdersEquity') / weighted_shares
    
    def calculate_net_operate_cash_flow_per_share(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        
        return self.safe_get_value_ttm(self.cash_flow_data, 'operatingCashFlow') / weighted_shares

    def calculate_eps(self):
        return self.safe_get_value_ttm(self.income_data, 'eps')

    def calculate_retained_earnings_per_share(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        return self.safe_get_value(self.balance_data, 'retainedEarnings') / weighted_shares
    
    def calculate_freecashflow_per_share(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        return self.safe_get_value_ttm(self.cash_flow_data, 'freeCashFlow') / weighted_shares
    
    def calculate_liquidity(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        
        return self.safe_get_value(self.market_data, 'volume') / weighted_shares

    def calculate_market_cap(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        close_price = self.safe_get_value(self.market_data, 'close')
        if pd.isna(weighted_shares) or pd.isna(close_price):
            return np.nan
        return weighted_shares * close_price

    def calculate_all_factors(self):
        factors = []
        for _, income_row in self.income_data_master.iterrows():
            try:
                date = income_row['date']
                self.income_data = self.income_data_master[self.income_data_master['date'] == date]
                self.balance_data = self.balance_data_master[self.balance_data_master['date'] == date]
                self.cash_flow_data = self.cash_flow_data_master[self.cash_flow_data_master['date'] == date]
                if date in self.market_data_master['date'].values:
                    self.market_data = self.market_data_master[self.market_data_master['date'] == date]
                else:
                    prev_dates = self.market_data_master[self.market_data_master['date'] < date]
                    if prev_dates.empty:
                        # Use a row of NaNs if no market data is available.
                        market_row = pd.Series({'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan, 'volume': np.nan})
                        self.market_data = pd.DataFrame([market_row])
                    else:
                        prev_date = prev_dates['date'].max()
                        self.market_data = self.market_data_master[self.market_data_master['date'] == prev_date]
                # For income data:
                income_subset = self.income_data_master[self.income_data_master['date'] < date]
                if not income_subset.empty:
                    available = income_subset.iloc[-4:]
                    if len(available) < 4:
                        num_missing = 4 - len(available)
                        pad = pd.DataFrame({col: [np.nan] * num_missing for col in income_subset.columns})
                        self.prev_income_data = pd.concat([pad, available], ignore_index=True)
                    else:
                        self.prev_income_data = available
                else:
                    self.prev_income_data = None

                # For balance data:
                balance_subset = self.balance_data_master[self.balance_data_master['date'] < date]
                if not balance_subset.empty:
                    available = balance_subset.iloc[-4:]
                    if len(available) < 4:
                        num_missing = 4 - len(available)
                        pad = pd.DataFrame({col: [np.nan] * num_missing for col in balance_subset.columns})
                        self.prev_balance_data = pd.concat([pad, available], ignore_index=True)
                    else:
                        self.prev_balance_data = available
                else:
                    self.prev_balance_data = None

                # For cash flow data:
                cash_flow_subset = self.cash_flow_data_master[self.cash_flow_data_master['date'] < date]
                if not cash_flow_subset.empty:
                    available = cash_flow_subset.iloc[-4:]
                    if len(available) < 4:
                        num_missing = 4 - len(available)
                        pad = pd.DataFrame({col: [np.nan] * num_missing for col in cash_flow_subset.columns})
                        self.prev_cash_flow_data = pd.concat([pad, available], ignore_index=True)
                    else:
                        self.prev_cash_flow_data = available
                else:
                    self.prev_cash_flow_data = None

                factors.append({
                    'date': date,
                    'open': self.safe_get_value(self.market_data, 'open'),
                    'high': self.safe_get_value(self.market_data, 'high'),
                    'low': self.safe_get_value(self.market_data, 'low'),
                    'close': self.safe_get_value(self.market_data, 'close'),
                    'volume': self.safe_get_value(self.market_data, 'volume'),
                    'net_asset_per_share': self.calculate_net_asset_per_share(),
                    'net_operate_cash_flow_per_share': self.calculate_net_operate_cash_flow_per_share(),
                    'eps': self.calculate_eps(),
                    'retained_earnings_per_share': self.calculate_retained_earnings_per_share(),
                    'cashflow_per_share': self.calculate_freecashflow_per_share(),
                    'liquidity': self.calculate_liquidity(),
                    'market_cap': self.calculate_market_cap()
                })
            except Exception as e:
                print(f"Error calculating stock factors for date {income_row['date']}: {e}")
        try:
            return pd.DataFrame(factors)
        except Exception as e:
            print(f"Error compiling stock factors: {e}")
            return pd.DataFrame()
