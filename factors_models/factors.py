import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
from data_sources.fmp import FMPWrapper
from .quality import Quality
from .value import Value
from .stock import Stock
from .growth import Growth
from .emotional import Emotional
from .style import Style
from .risk import Risk
from .momentum import Momentum
from .technical import Technical

class FactorsWrapper:
    """
    A wrapper class to aggregate all factor calculations for a given ticker using data from FMP.

    It fetches financial statements and market data via the FMPWrapper, converts the responses to DataFrames,
    and passes them to the individual factor calculation classes.
    """
    def __init__(self, ticker, fmp: FMPWrapper, start_date, end_date, period="quarterly"):
        self.ticker = ticker
        self.fmp = fmp
        self.period = period
        self.start_date = start_date
        self.end_date = end_date

        # Fetch current financial statements and convert to DataFrames.
        self.income_data = self._get_df(self.fmp.get_income_statement(self.ticker, period=self.period))
        self.balance_data = self._get_df(self.fmp.get_balance_sheet(self.ticker, period=self.period))
        self.cash_flow_data = self._get_df(self.fmp.get_cash_flow_statement(self.ticker, period=self.period))
        self.financial_ratio_data = self._get_df(self.fmp.get_financial_ratios(self.ticker, period=self.period))

        self.prev_quarter_start_date = self.get_prev_quarter_start(start_date)

        self.balance_data = self.balance_data[(self.balance_data['date'] >= self.prev_quarter_start_date) & (self.balance_data['date'] <= self.end_date)][['date', 'symbol', 'calendarYear', 'period', 'netReceivables', 'inventory', 'totalAssets', 'totalLiabilities', 'totalDebt','minorityInterest', 'commonStock', 'totalStockholdersEquity', 'retainedEarnings', "accountPayables", 'totalCurrentAssets', 'totalCurrentLiabilities']].iloc[::-1].reset_index(drop=True)
        self.income_data = self.income_data[(self.income_data['date'] >= self.prev_quarter_start_date) & (self.income_data['date'] <= self.end_date)][['date', 'symbol','calendarYear', 'period', 'revenue', 'grossProfit', 'netIncome', 'interestExpense', 'eps', 'operatingExpenses', 'costOfRevenue', 'operatingIncome','weightedAverageShsOut']].iloc[::-1].reset_index(drop=True)
        self.cash_flow_data = self.cash_flow_data[(self.cash_flow_data['date'] >= self.prev_quarter_start_date) & (self.cash_flow_data['date'] <= self.end_date)][['date', 'symbol','calendarYear', 'period', 'dividendsPaid', 'operatingCashFlow', 'freeCashFlow']].iloc[::-1].reset_index(drop=True)
        self.financial_ratio_data = self.financial_ratio_data[(self.financial_ratio_data['date'] >= self.prev_quarter_start_date) & (self.financial_ratio_data['date'] <= self.end_date)].iloc[::-1].reset_index(drop=True)

        # print(self.balance_data)

        # Fetch historical market data for market-related factors.
        self.market_data = self.fmp.get_historical_price(self.ticker, self.prev_quarter_start_date, end_date)
        self.market_data = self.market_data[['date','open', 'high', 'low', 'close', 'adjClose', 'volume', 'changePercent']].iloc[::-1].reset_index(drop=True)
        # print(self.market_data)

    def get_prev_quarter_start(self, date_str):
        date = pd.to_datetime(date_str)
        # Get current quarter
        current_quarter = (date.month - 1) // 3
        # Calculate previous quarter
        prev_quarter = (current_quarter - 1) % 4
        # Calculate year adjustment if moving back to previous year
        year_adj = -1 if current_quarter == 0 else 0
        # Get first month of previous quarter
        prev_quarter_month = prev_quarter * 3 + 1
        # Create previous quarter start date
        prev_quarter_start = pd.Timestamp(
            year=date.year + year_adj,
            month=prev_quarter_month,
            day=1
        )
        return prev_quarter_start.strftime('%Y-%m-%d')

    def _get_df(self, data):
        """
        Convert the JSON response (a list of dicts) from FMPWrapper to a pandas DataFrame.
        If the list contains more than one period, use the element at the provided index.
        """
        if isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()

    def calculate_all_factors(self):
        results = {}

        # Quality factors
        try:
            quality_obj = Quality(
                income_data=self.income_data,
                balance_data=self.balance_data,
                cash_flow_data=self.cash_flow_data,
            )
            results['quality'] = quality_obj.calculate_all_factors()
        except Exception as e:
            results['quality'] = f"Error: {e}"

        # Value factors
        try:
            value_obj = Value(
                income_data=self.income_data,
                balance_data=self.balance_data,
                cash_flow_data=self.cash_flow_data,
                market_data = self.market_data,
                financial_ratio_data=self.financial_ratio_data
            )
            results['value'] = value_obj.calculate_all_factors()
        except Exception as e:
            results['value'] = f"Error: {e}"

        # # Stock factors
        try:
            stock_obj = Stock(
                income_data=self.income_data,
                balance_data=self.balance_data,
                cash_flow_data=self.cash_flow_data,
                market_data = self.market_data
            )
            results['stock'] = stock_obj.calculate_all_factors()
        except Exception as e:
            results['stock'] = f"Error: {e}"

        # # Growth factors
        try:
            growth_obj = Growth(
                income_data=self.income_data,
                balance_data=self.balance_data,
                cash_flow_data=self.cash_flow_data,
                market_data=self.market_data
            )
            results['growth'] = growth_obj.calculate_all_factors()
        except Exception as e:
            results['growth'] = f"Error: {e}"

        # Emotional factors using market data
        try:
            emotional_obj = Emotional(self.market_data)
            results['emotional'] = emotional_obj.calculate_all_factors()
        except Exception as e:
            results['emotional'] = f"Error: {e}"

        # Style factors (stub - adjust if style factor calculations are implemented)
        combined_data = self.market_data.copy()
        tickers = [self.ticker]
        sp_500 = self.fmp.get_historical_price('^GSPC', self.prev_quarter_start_date, self.end_date)
        try:
            style_obj = Style(combined_data, sp_500[['date','changePercent']], tickers)
            results['style'] = style_obj.calculate_all_factors()
        except Exception as e:
            results['style'] = f"Error: {e}"

        # Risk factors
        try:
            risk_free_rate_annual = 0.065
            risk_obj = Risk(
                df=self.market_data,
                risk_free_rate_20=(1 + risk_free_rate_annual) ** (20 / 252) - 1,
                risk_free_rate_60=(1 + risk_free_rate_annual) ** (60 / 252) - 1
            )
            results['risk'] = risk_obj.calculate_all_factors()
        except Exception as e:
            results['risk'] = f"Error: {e}"

        # Momentum factors
        try:
            momentum_obj = Momentum(self.market_data)
            results['momentum'] = momentum_obj.calculate_all_factors()
        except Exception as e:
            results['momentum'] = f"Error: {e}"

        # Technical factors
        try:
            technical_obj = Technical(self.market_data)
            results['technical'] = technical_obj.calculate_all_factors()
        except Exception as e:
            results['technical'] = f"Error: {e}"

        return results

if __name__ == "__main__":
    # For testing this module individually.
    api_key = "bEiVRux9rewQy16TXMPxDqBAQGIW8UBd"
    fmp = FMPWrapper(api_key)
    ticker = "AAPL"
    start_date = "1995-01-01"
    end_date = "2024-12-31"
    wrapper = FactorsWrapper(ticker, fmp, start_date, end_date)
    all_factors = wrapper.calculate_all_factors()
    print(all_factors)
