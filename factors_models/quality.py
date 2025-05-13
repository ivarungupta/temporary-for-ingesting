import pandas as pd
import numpy as np

class Quality:
    """
    A class to calculate quality-related financial factors.
    """
    def __init__(self, income_data, balance_data, cash_flow_data):
        self.income_data_master = income_data
        self.balance_data_master = balance_data
        self.cash_flow_data_master = cash_flow_data
        self.required_columns = {
            'income': {'netIncome', 'revenue', 'grossProfit'},
            'balance': {'totalStockholdersEquity', 'totalAssets', 'totalDebt', 'totalLiabilities', 'inventory', "accountPayables"},
            'cash_flow': {'operatingCashFlow'}
        }
        self._validate_columns()

    def _validate_columns(self):
        missing_cols = []
        for col in self.required_columns['income']:
            if col not in self.income_data_master.columns:
                missing_cols.append(f"Income: {col}")
        for col in self.required_columns['balance']:
            if col not in self.balance_data_master.columns:
                missing_cols.append(f"Balance: {col}")
        for col in self.required_columns['cash_flow']:
            if col not in self.cash_flow_data_master.columns:
                missing_cols.append(f"Cash Flow: {col}")
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    def safe_get_value_ttm(self, df, column):
        if df.empty or column not in df.columns:
            return np.nan
        try:
            # If fewer than 4 rows exist, sum what is available
            if len(df) < 4:
                return df[column].iloc[:len(df)].sum()
            # Sum the values of the first 4 rows (most recent four quarters)
            return df[column].iloc[:4].sum()
        except Exception:
            return np.nan
        
    def calculate_net_profit_to_revenue(self):
        if self.income_data is None:
            return np.nan
        
        revenue_ttm = self.safe_get_value_ttm(self.income_data, 'revenue')
        if pd.isna(revenue_ttm) or revenue_ttm == 0:
            return np.nan

        # Get TTM net income.
        net_income_ttm = self.safe_get_value_ttm(self.income_data, 'netIncome')
        return net_income_ttm / revenue_ttm

    def calculate_decm(self):
        # For previous data, check for None or empty before accessing .iloc[0]
        if self.prev_balance_data is None or self.prev_balance_data.empty or self.prev_balance_data['inventory'].iloc[0] == 0:
            prev_inventory = 0
        else:
            prev_inventory = self.prev_balance_data['inventory'].iloc[0]
        
        if self.prev_balance_data is None or self.prev_balance_data.empty or self.prev_balance_data['accountPayables'].iloc[0] == 0:
            prev_acc_payables = 0
        else:
            prev_acc_payables = self.prev_balance_data['accountPayables'].iloc[0]

        if self.balance_data is None or self.balance_data.empty or self.balance_data['inventory'].iloc[0] == 0:
            curr_inventory = 0
        else:
            curr_inventory = self.balance_data['inventory'].iloc[0]
        
        if self.balance_data is None or self.balance_data.empty or self.balance_data['accountPayables'].iloc[0] == 0:
            curr_acc_payables = 0
        else:
            curr_acc_payables = self.balance_data['accountPayables'].iloc[0]
        
        if self.balance_data is None or self.balance_data.empty or self.balance_data['totalAssets'].iloc[0] == 0:
            return np.nan
        
        decm = curr_acc_payables + curr_inventory - prev_acc_payables - prev_inventory 
        return decm / self.balance_data['totalAssets'].iloc[0]

    def calculate_roe(self):
        if self.balance_data is None or self.balance_data.empty or self.balance_data['totalStockholdersEquity'].iloc[0] == 0:
            return np.nan
        
        net_income_ttm = self.safe_get_value_ttm(self.income_data, 'netIncome')
        return net_income_ttm / self.balance_data['totalStockholdersEquity'].iloc[0]

    def calculate_roa(self):
        if self.prev_balance_data is None or self.prev_balance_data.empty or self.prev_balance_data['totalAssets'].iloc[0] == 0:
            return np.nan
        
        if self.balance_data is None or self.balance_data.empty or self.balance_data['totalAssets'].iloc[0] == 0:
            return np.nan
        
        net_income_ttm = self.safe_get_value_ttm(self.income_data, 'netIncome')
        return net_income_ttm / (self.balance_data['totalAssets'].iloc[0] + self.prev_balance_data['totalAssets'].iloc[0])

    def calculate_gmi(self):
        if self.prev_income_data is None or self.prev_income_data.empty or self.prev_income_data['revenue'].iloc[0] == 0:
            prev_gross_margin = 0
        else:
            prev_gross_margin = self.prev_income_data['grossProfit'].iloc[0] / self.prev_income_data['revenue'].iloc[0]
        
        if self.income_data is None or self.income_data.empty or self.income_data['revenue'].iloc[0] == 0:
            return np.nan
        
        current_gross_margin = self.income_data['grossProfit'].iloc[0] / self.income_data['revenue'].iloc[0]
        return current_gross_margin - prev_gross_margin

    def calculate_acca(self):
        if self.balance_data is None or self.balance_data.empty or self.balance_data['totalAssets'].iloc[0] == 0:
            return np.nan

        net_income_ttm = self.safe_get_value_ttm(self.income_data, 'netIncome')
        operating_cf_ttm = self.safe_get_value_ttm(self.cash_flow_data, 'operatingCashFlow')
        return (net_income_ttm - operating_cf_ttm) / self.balance_data['totalAssets'].iloc[0]

    def calculate_debtToAsset(self):
        if self.balance_data is None or self.balance_data.empty or self.balance_data['totalAssets'].iloc[0] == 0:
            return np.nan
        return self.balance_data['totalDebt'].iloc[0] / self.balance_data['totalAssets'].iloc[0]
    
    def calculate_all_factors(self):
        try:
            factors = []
            # Loop over each row (each date) in the master income data.
            for _, income_row in self.income_data_master.iterrows():
                date = income_row['date']
                self.income_data = self.income_data_master[self.income_data_master['date'] == date]
                self.balance_data = self.balance_data_master[self.balance_data_master['date'] == date]
                self.cash_flow_data = self.cash_flow_data_master[self.cash_flow_data_master['date'] == date]

                # For previous income data:
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

                # For previous balance data:
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

                # For previous cash flow data:
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
                    'net_profit_to_total_revenue': self.calculate_net_profit_to_revenue(),
                    'DECM': self.calculate_decm(),
                    'ROE': self.calculate_roe(),
                    'ROA': self.calculate_roa(),
                    'ACCA': self.calculate_acca(),
                    'GMI': self.calculate_gmi(),
                    'DtoA': self.calculate_debtToAsset()
                })
                
            return pd.DataFrame(factors)
        except Exception as e:
            print(f"Error calculating quality factors: {e}")
            return pd.DataFrame()
