import pandas as pd
import numpy as np

class Growth:
    """
    A class to calculate growth-related financial factors.
    """
    def __init__(self, income_data, balance_data, cash_flow_data, market_data):
        self.income_data_master = income_data
        self.balance_data_master = balance_data
        self.cash_flow_data_master = cash_flow_data
        self.market_data_master = market_data

        self.required_columns = {
            'income': {'eps', 'netIncome', 'revenue'},
            'balance': {'totalStockholdersEquity'},
            'cash_flow': {'operatingCashFlow'},
            'market' : {'close'}
        }
        self._validate_columns()

    def _validate_columns(self):
        missing_cols = []
        for df_name, columns in self.required_columns.items():
            current_df = getattr(self, f"{df_name}_data_master")
            for col in columns:
                if col not in current_df.columns:
                    missing_cols.append(f"Current {df_name}: {col}")
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
        
    def safe_get_value_ttm_3(self, df, column):
        if df.empty or column not in df.columns:
            return np.nan
        try:
            # Ensure there are at least 3 quarters available
            if len(df) < 3:
                return df[column].iloc[:len(df)].sum()
            # Sum the values of the first 3 rows (most recent four quarters)
            return df[column].iloc[:3].sum()
        
        except Exception:
            return np.nan
        
    def calculate_peg(self):
        # Get the current close price.
        close_price = self.safe_get_value(self.market_data, 'close')

        # Retrieve previous trailing EPS (for 3 periods) if available.
        prev_eps_ttm_3 = (
            self.safe_get_value_ttm_3(self.prev_income_data, 'eps')
            if self.prev_income_data is not None else np.nan
        )

        # If we don't have a valid previous EPS or it's zero, return nan.
        if pd.isna(prev_eps_ttm_3) or prev_eps_ttm_3 == 0:
            return np.nan

        # Get current EPS.
        current_eps = self.safe_get_value(self.income_data, 'eps')
        # Calculate total trailing EPS.
        eps_ttm = current_eps + prev_eps_ttm_3

        # Retrieve previous EPS if available.
        prev_eps = (
            self.safe_get_value(self.prev_income_data, 'eps')
            if self.prev_income_data is not None else np.nan
        )

        # Calculate EPS growth if possible.
        if pd.isna(prev_eps) or prev_eps == 0:
            eps_growth = np.nan
        else:
            eps_growth = (current_eps / prev_eps) - 1

        # Check to avoid division by zero if growth is 0 or undefined.
        if pd.isna(eps_growth) or eps_growth == 0:
            return np.nan
        if pd.isna(eps_ttm) or eps_ttm == 0:
            return np.nan   
        # Calculate PEG ratio.
        peg = (close_price / eps_ttm) / abs(eps_growth)
        return peg


    def calculate_net_profit_growth(self):
        prev_net = self.safe_get_value(self.prev_income_data, 'netIncome') if self.prev_income_data is not None else np.nan
        if pd.isna(prev_net) or prev_net == 0:
            return np.nan
        return (self.safe_get_value(self.income_data, 'netIncome') / prev_net) - 1
    
    def calculate_revenue_growth(self):
        prev_rev = self.safe_get_value(self.prev_income_data, 'revenue') if self.prev_income_data is not None else np.nan
        if pd.isna(prev_rev) or prev_rev == 0:
            return np.nan
        return (self.safe_get_value(self.income_data, 'revenue') / prev_rev) - 1

    def calculate_net_asset_growth(self):
        prev_equity = self.safe_get_value(self.prev_balance_data, 'totalStockholdersEquity') if self.prev_balance_data is not None else np.nan
        if pd.isna(prev_equity) or prev_equity == 0:
            return np.nan
        return (self.safe_get_value(self.balance_data, 'totalStockholdersEquity') / prev_equity) - 1

    def calculate_operating_cashflow_growth(self):
        prev_ocf = self.safe_get_value(self.prev_cash_flow_data, 'operatingCashFlow') if self.prev_cash_flow_data is not None else np.nan
        if pd.isna(prev_ocf) or prev_ocf == 0:
            return np.nan
        return (self.safe_get_value(self.cash_flow_data, 'operatingCashFlow') / prev_ocf) - 1

    # eps growth rate
    def calculate_eps_growth_rate(self):
        prev_eps_ttm = (
            self.safe_get_value_ttm(self.prev_income_data, 'eps')
            if self.prev_income_data is not None else np.nan
        )
        if pd.isna(prev_eps_ttm) or prev_eps_ttm == 0:
            return np.nan

        # Retrieve the previous trailing EPS for three periods
        prev_eps_ttm_3 = (
            self.safe_get_value_ttm_3(self.prev_income_data, 'eps')
            if self.prev_income_data is not None else np.nan
        )

        if pd.isna(prev_eps_ttm_3) or prev_eps_ttm_3 == 0:
            return np.nan
        
        # Retrieve the current EPS from the income data
        current_eps = self.safe_get_value(self.income_data, 'eps')

        # Calculate and return the EPS growth rate
        return ((current_eps + prev_eps_ttm_3) / prev_eps_ttm) - 1
    
    def calculate_all_factors(self):
        factors = []
        for _, income_row in self.income_data_master.iterrows():
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
                'PEG': self.calculate_peg(),
                'net_profit_growth_rate': self.calculate_net_profit_growth(),
                'total_revenue_growth_rate': self.calculate_revenue_growth(),
                'net_asset_growth_rate': self.calculate_net_asset_growth(),
                'net_operate_cashflow_growth_rate': self.calculate_operating_cashflow_growth()
            })
        try:
            return pd.DataFrame(factors)
        except Exception as e:
            print(f"Error calculating growth factors: {e}")
            return pd.DataFrame()

