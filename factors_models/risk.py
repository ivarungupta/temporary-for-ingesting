import pandas as pd

class Risk:
    """
    A class to calculate risk-related market factors.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing market data
        risk_free_rate_20 (float): 20-day risk-free rate
        risk_free_rate_60 (float): 60-day risk-free rate
    """
    def __init__(self, df, risk_free_rate_20=0.02, risk_free_rate_60=0.02):
        self.df = df
        self.risk_free_rate_20 = risk_free_rate_20
        self.risk_free_rate_60 = risk_free_rate_60
        self.required_columns = {'close'}
        self._validate_columns()

    def _validate_columns(self):
        missing_cols = self.required_columns - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def calculate_variance(self, window=60):
        self.df['Variance60'] = self.df['close'].pct_change().rolling(window=window).var()
        return self.df

    def calculate_sharpe_ratio_20(self):
        returns = self.df['close'].pct_change()
        self.df['sharpe_ratio_20'] = (returns.rolling(window=20).mean() - self.risk_free_rate_20) / returns.rolling(window=20).std()
        return self.df

    def calculate_kurtosis(self, window=60):
        self.df['Kurtosis60'] = self.df['close'].pct_change().rolling(window=window).kurt()
        return self.df

    def calculate_skewness(self, window=60):
        self.df['Skewness60'] = self.df['close'].pct_change().rolling(window=window).skew()
        return self.df

    def calculate_sharpe_ratio_60(self):
        returns = self.df['close'].pct_change()
        self.df['sharpe_ratio_60'] = (returns.rolling(window=60).mean() - self.risk_free_rate_60) / returns.rolling(window=60).std()
        return self.df

    def calculate_all_factors(self):
        try:
            self.calculate_variance()
            self.calculate_sharpe_ratio_20()
            self.calculate_kurtosis()
            self.calculate_skewness()
            self.calculate_sharpe_ratio_60()
            risk_columns = ['date','Variance60', 'sharpe_ratio_20', 'Kurtosis60', 'Skewness60', 'sharpe_ratio_60']
            return self.df[risk_columns]
        except Exception as e:
            print(f"Error calculating risk factors: {e}")
            return pd.DataFrame()
