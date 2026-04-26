# user_data/strategies/CatBoostStrategy.py
import pandas as pd
from functools import reduce
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class CatBoostStrategy(IStrategy):
    """
    Phase 1: Algorithmic Core - CatBoost Model for High-Frequency Microstructure
    """
    timeframe = "5m"
    can_short = True
    startup_candle_count = 100

    # Mandatory attributes
    stoploss = -0.10
    minimal_roi = {
        "0": 0.05
    }

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-volume-change"] = dataframe["volume"].pct_change()
        
        # Microstructure features integration (Phase 1)
        dataframe["%-spread"] = dataframe["high"] - dataframe["low"]
        dataframe["%-price-range"] = (dataframe["close"] - dataframe["low"]) / (dataframe["high"] - dataframe["low"] + 0.0001)
        
        # Mocking Cumulative Volume Delta (CVD) as volume * price direction
        dataframe["%-mock-cvd"] = dataframe["volume"] * (dataframe["close"] - dataframe["open"])
        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Target: predict 3 candles into the future
        dataframe["&-s_close"] = (
            dataframe["close"].shift(-3) / dataframe["close"] - 1
        )
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        enter_long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-s_close"] > 0.01  # Predicted to go up > 1%
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "freqai_long")

        enter_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-s_close"] < -0.01 # Predicted to go down > 1%
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "freqai_short")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        exit_long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-s_close"] < 0
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-s_close"] > 0
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return dataframe
