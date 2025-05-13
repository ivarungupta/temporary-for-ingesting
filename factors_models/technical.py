import pandas as pd

class Technical:
    """
    A class to calculate technical market factors.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing OHLCV market data
    """
    def __init__(self, df):
        self.df = df
        self.required_columns = {'close', 'high', 'low', 'volume'}
        self._validate_columns()

    def _validate_columns(self):
        missing_cols = self.required_columns - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def calculate_mac(self, fast_span=36, slow_span=78):
        fast_ma = self.df['close'].ewm(span=fast_span).mean()
        slow_ma = self.df['close'].ewm(span=slow_span).mean()
        self.df['MAC60'] = fast_ma - slow_ma
        return self.df

    def calculate_bollinger_bands(self, window=60, num_std=2):
        rolling_mean = self.df['close'].rolling(window=window).mean()
        rolling_std = self.df['close'].rolling(window=window).std()
        self.df['boll_up'] = rolling_mean + (rolling_std * num_std)
        self.df['boll_down'] = rolling_mean - (rolling_std * num_std)
        return self.df

    def calculate_mfi(self, window=42):
        typical_price = (self.df['close'] + self.df['high'] + self.df['low']) / 3
        raw_money_flow = typical_price * self.df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_flow_sum = positive_flow.rolling(window=window).sum()
        negative_flow_sum = negative_flow.rolling(window=window).sum()
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        self.df['MFI42'] = 100 - (100 / (1 + money_flow_ratio))
        return self.df

    def calculate_all_factors(self):
        try:
            self.calculate_mac()
            self.calculate_bollinger_bands()
            self.calculate_mfi()
            technical_columns = ['date','MAC60', 'boll_up', 'boll_down', 'MFI42']
            return self.df[technical_columns]
        except Exception as e:
            print(f"Error calculating technical factors: {e}")
            return pd.DataFrame()
