import pandas as pd

class Momentum:
    """
    A class to calculate momentum-related market factors.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing OHLCV market data
    """
    def __init__(self, df):
        self.df = df
        self.required_columns = {'close', 'volume'}
        self._validate_columns()

    def _validate_columns(self):
        missing_cols = self.required_columns - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def calculate_rate_of_change(self, window=60):
        self.df['ROC60'] = (self.df['close'] - self.df['close'].shift(window)) / self.df['close'].shift(window)
        return self.df

    def calculate_volume_quarterly(self, window=60):
        self.df['Volume1Q'] = self.df['volume'].rolling(window=window).sum()
        return self.df

    def calculate_trix(self, span=30):
        triple_ema = self.df['close'].ewm(span=span).mean().ewm(span=span).mean().ewm(span=span).mean()
        self.df['TRIX30'] = triple_ema.pct_change(periods=1)
        return self.df

    def calculate_price_quarterly(self, window=60):
        self.df['Price1Q'] = self.df['close'] - self.df['close'].shift(window)
        return self.df

    def calculate_price_level_ratio(self, window=36):
        self.df['PLRC36'] = self.df['close'].rolling(window=window).mean() / self.df['close'].shift(window) - 1
        return self.df

    def calculate_all_factors(self):
        try:
            self.calculate_rate_of_change()
            self.calculate_volume_quarterly()
            self.calculate_trix()
            self.calculate_price_quarterly()
            self.calculate_price_level_ratio()
            momentum_columns = ['date','ROC60', 'Volume1Q', 'TRIX30', 'Price1Q', 'PLRC36']
            return self.df[momentum_columns]
        except Exception as e:
            print(f"Error calculating momentum factors: {e}")
            return pd.DataFrame()
