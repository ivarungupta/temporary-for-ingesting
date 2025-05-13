import pandas as pd

class Emotional:
    """
    A class to calculate emotional/sentiment-related market factors.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing market data with OHLCV columns
    """
    def __init__(self, df):
        self.df = df
        self.required_columns = {'close', 'high', 'low', 'volume'}
        self._validate_columns()

    def _validate_columns(self):
        missing_cols = self.required_columns - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def calculate_volume_volatility(self, window=60):
        self.df['VOL60'] = self.df['close'].pct_change().rolling(window=window).std()
        return self.df

    def calculate_volume_ma(self, window=60):
        self.df['DAVOL60'] = self.df['volume'].rolling(window=window).mean()
        return self.df

    def calculate_volume_oscillator(self):
        volume_ma = self.df['volume'].rolling(window=60).mean()
        self.df['VOSC'] = self.df['volume'] - volume_ma
        return self.df

    def calculate_volume_macd(self):
        fast_ma = self.df['volume'].ewm(span=36).mean()
        slow_ma = self.df['volume'].ewm(span=78).mean()
        self.df['VMACD'] = fast_ma - slow_ma
        return self.df

    def calculate_atr(self, window=42):
        self.df['ATR42'] = (self.df['high'] - self.df['low']).rolling(window=window).mean()
        return self.df

    def calculate_all_factors(self):
        try:
            self.calculate_volume_volatility()
            self.calculate_volume_ma()
            self.calculate_volume_oscillator()
            self.calculate_volume_macd()
            self.calculate_atr()
            emotional_columns = ['date','VOL60', 'DAVOL60', 'VOSC', 'VMACD', 'ATR42']
            return self.df[emotional_columns]
        except Exception as e:
            print(f"Error calculating emotional factors: {e}")
            return pd.DataFrame()
