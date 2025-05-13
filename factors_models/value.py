import pandas as pd
import numpy as np

class Value:
    """
    A class to calculate value-related financial factors.
    """
    def __init__(self, income_data, balance_data, cash_flow_data, market_data, financial_ratio_data):
        self.income_data_master = income_data
        self.balance_data_master = balance_data
        self.cash_flow_data_master = cash_flow_data
        self.market_data_master = market_data
        self.financial_ratio_data_master = financial_ratio_data
        self.required_columns = {
            'income': {'netIncome', 'grossProfit', 'revenue', 'eps', 'costOfRevenue', 'operatingExpenses', 'weightedAverageShsOut'},
            'balance': {'totalLiabilities', 'totalAssets', 'netReceivables', 'inventory', 'totalStockholdersEquity', 'totalCurrentAssets', 'totalCurrentLiabilities'},
            'cash_flow': {'operatingCashFlow'},
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

    def calculate_financial_liability(self):
        return self.safe_get_value(self.balance_data, 'totalLiabilities')

    def calculate_cashflow_price(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        close_price = self.safe_get_value(self.market_data, 'close')
        
        operating_cf_ttm = self.safe_get_value_ttm(self.cash_flow_data, 'operatingCashFlow')

        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        
        if pd.isna(close_price) or close_price == 0:
            return np.nan
        
        return (operating_cf_ttm / weighted_shares) / close_price
    
    def calculate_priceToBook(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        close_price = self.safe_get_value(self.market_data, 'close')
        stock_holders_eq = self.safe_get_value(self.balance_data, 'totalStockholdersEquity')

        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        
        if pd.isna(stock_holders_eq) or stock_holders_eq == 0:
            return np.nan
        
        return close_price/(stock_holders_eq/weighted_shares)

    def calculate_price_to_sales(self):
        weighted_shares = self.safe_get_value(self.income_data, 'weightedAverageShsOut')
        close_price = self.safe_get_value(self.market_data, 'close')
        revenue = self.safe_get_value_ttm(self.income_data, 'revenue')

        if pd.isna(weighted_shares) or weighted_shares == 0:
            return np.nan
        
        if pd.isna(revenue) or revenue == 0:
            return np.nan
        
        return close_price/(revenue/weighted_shares)

    def calculate_price_to_earnings(self):
        close_price = self.safe_get_value(self.market_data, 'close')
        eps = self.safe_get_value_ttm(self.income_data, 'eps')
        
        if pd.isna(eps) or eps == 0:
            return np.nan
        return close_price/eps

    def calculate_ltd_to_ta(self):
        total_assets = self.safe_get_value(self.balance_data, 'totalAssets')
        if pd.isna(total_assets) or total_assets == 0:
            return np.nan
        total_liabilities = self.safe_get_value(self.balance_data, 'totalLiabilities')
        return total_liabilities / total_assets
    
    def calculate_net_profit(self):
        return self.safe_get_value_ttm(self.income_data, 'netIncome')

    def calculate_ebit(self):
        g_profit = self.safe_get_value_ttm(self.income_data, 'grossProfit')
        op_exp = self.safe_get_value_ttm(self.income_data, 'operatingExpenses')

        if pd.isna(g_profit) or pd.isna(op_exp):
            return np.nan
        
        return g_profit - op_exp

    def calculate_working_capital_ratio(self):
        cassets = self.safe_get_value(self.balance_data, 'totalCurrentAssets')
        cliabi = self.safe_get_value(self.balance_data, 'totalCurrentLiabilities')
        if pd.isna(cassets) or pd.isna(cliabi):
            return np.nan
        return cassets - cliabi
    
    def calculate_quick_ratio(self):
        cassets = self.safe_get_value(self.balance_data, 'totalCurrentAssets')
        cliabi = self.safe_get_value(self.balance_data, 'totalCurrentLiabilities')
        if pd.isna(cliabi) or cliabi == 0:
            return np.nan
        inventory = self.safe_get_value(self.balance_data, 'inventory')
        if pd.isna(cassets) or pd.isna(inventory):
            return np.nan
        return (cassets - inventory) / cliabi

    # operating cashflow to total asset
    def calculate_operating_cashflow_to_total_assets(self):

        # Check that cash flow and balance data are available and non-empty.
        if not hasattr(self.cash_flow_data, 'empty') or self.cash_flow_data.empty:
            return np.nan
        if not hasattr(self.balance_data, 'empty') or self.balance_data.empty:
            return np.nan

        # Retrieve operating cash flow and total assets.
        operating_cashflow = self.safe_get_value(self.cash_flow_data, 'operatingCashFlow')
        total_assets = self.safe_get_value(self.balance_data, 'totalAssets')

        # Avoid division by zero.
        if pd.isna(total_assets) or total_assets == 0:
            return np.nan

        return operating_cashflow / total_assets


    # ev to operating cashflow (ttm)
    def calculate_ev_to_operating_cashflow(self):

        # Check that market and cash flow data are available.
        if not hasattr(self.market_data, 'empty') or self.market_data.empty:
            return np.nan
        if not hasattr(self.cash_flow_data, 'empty') or self.cash_flow_data.empty:
            return np.nan

        # Retrieve Enterprise Value from market data.
        ev = self.safe_get_value(self.market_data, 'enterpriseValue')

        # Retrieve TTM operating cash flow.
        operating_cashflow_ttm = self.safe_get_value_ttm(self.cash_flow_data, 'operatingCashFlow')

        # Avoid division by zero.
        if pd.isna(operating_cashflow_ttm) or operating_cashflow_ttm == 0:
            return np.nan

        return ev / operating_cashflow_ttm


    # operating cashflow to net profit
    def calculate_operating_cashflow_to_net_profit(self):
        # Ensure cash flow and income data are available.
        if not hasattr(self.cash_flow_data, 'empty') or self.cash_flow_data.empty:
            return np.nan
        if not hasattr(self.income_data, 'empty') or self.income_data.empty:
            return np.nan

        # Retrieve TTM operating cash flow and TTM net profit.
        operating_cashflow_ttm = self.safe_get_value_ttm(self.cash_flow_data, 'operatingCashFlow')
        net_profit_ttm = self.safe_get_value_ttm(self.income_data, 'netIncome')

        # Avoid division by zero.
        if pd.isna(net_profit_ttm) or net_profit_ttm == 0:
            return np.nan

        return operating_cashflow_ttm / net_profit_ttm


    # debt to ebitda
    def calculate_debt_to_ebitda(self):
        # Ensure balance and income data are available.
        if not hasattr(self.balance_data, 'empty') or self.balance_data.empty:
            return np.nan
        if not hasattr(self.income_data, 'empty') or self.income_data.empty:
            return np.nan
        
        # Retrieve total debt from the balance data if available; otherwise, compute as total assets - total equity.
        if 'totalDebt' in self.balance_data.columns:
            total_debt = self.safe_get_value(self.balance_data, 'totalDebt')
        else:
            total_debt = self.safe_get_value(self.balance_data, 'totalAssets') - self.safe_get_value(self.balance_data, 'totalStockholdersEquity')

        # Retrieve TTM EBITDA.
        ebitda_ttm = self.safe_get_value_ttm(self.income_data, 'ebitda')

        # Avoid division by zero.
        if pd.isna(ebitda_ttm) or ebitda_ttm == 0:
            return np.nan

        return total_debt / ebitda_ttm

    
    def calculate_debt_to_equity(self):
        total_equity = self.safe_get_value(self.balance_data, 'totalStockholdersEquity')
        if pd.isna(total_equity) or total_equity == 0:
            return np.nan
        total_liabilities = self.safe_get_value(self.balance_data, 'totalLiabilities')
        return total_liabilities / total_equity

    def calculate_all_factors(self):
        factors = []
        for _, income_row in self.income_data_master.iterrows():
            date = income_row['date']
            self.income_data = self.income_data_master[self.income_data_master['date'] == date]
            self.balance_data = self.balance_data_master[self.balance_data_master['date'] == date]
            self.cash_flow_data = self.cash_flow_data_master[self.cash_flow_data_master['date'] == date]
            self.market_data = self.market_data_master[self.market_data_master['date'] == date]
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

            if self.income_data.empty or self.balance_data.empty or self.cash_flow_data.empty:
                print(f"Warning: Skipping value factors for date {date} due to missing data")
                continue

            factors.append({
                'date': date,
                'financial_liability': self.calculate_financial_liability(),
                'net_profit': self.calculate_net_profit(),
                'EBIT': self.calculate_ebit(),
                'LTD/TA': self.calculate_ltd_to_ta(),
                'WCR': self.calculate_working_capital_ratio(),
                'QR': self.calculate_quick_ratio(),
                'D/E': self.calculate_debt_to_equity(),
                'P/E': self.calculate_price_to_earnings(),
                'P/S': self.calculate_price_to_sales(),
                'CashFlowToPrice': self.calculate_cashflow_price(),
                'priceToBook': self.calculate_priceToBook(),
                'OpCashFlowToAssets': self.calculate_operating_cashflow_to_total_assets(),
                'Debt_Ebitda': self.calculate_debt_to_ebitda(),
                'EV/OCF': self.calculate_ev_to_operating_cashflow(),
                'OCF/NP': self.calculate_operating_cashflow_to_net_profit()
            })
        try:
            return pd.DataFrame(factors)
        except Exception as e:
            print(f"Error calculating value factors: {e}")
            return pd.DataFrame()
