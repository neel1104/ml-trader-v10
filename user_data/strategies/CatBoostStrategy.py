# user_data/strategies/CatBoostStrategy.py
import pandas as pd
from functools import reduce
from freqtrade.strategy import IStrategy, DecimalParameter
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

    # Hyperopt parameters
    buy_threshold = DecimalParameter(0.0005, 0.02, default=0.001, space='buy')
    sell_threshold = DecimalParameter(-0.02, -0.0005, default=-0.001, space='sell')

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Phase 3: Smart Money Features (Tick-level)
        metadata = kwargs.get('metadata', {})
        if self.dp and metadata:
            pair = metadata.get('pair')
            trades = self.dp.trades(pair)
            if trades is not None and not trades.empty:
                trades_df = trades.set_index('date')

                # Calculate True OFI (Buy Volume - Sell Volume)
                buy_vol = trades_df[trades_df['side'] == 'buy']['amount'].resample(self.timeframe).sum()
                sell_vol = trades_df[trades_df['side'] == 'sell']['amount'].resample(self.timeframe).sum()
                ofi = buy_vol.sub(sell_vol, fill_value=0)

                # Calculate Whale Proxy (Volume of top 5% of trades)
                whale_thresh = trades_df['amount'].quantile(0.95) if not trades_df.empty else 0
                whale_vol = trades_df[trades_df['amount'] > whale_thresh]['amount'].resample(self.timeframe).sum()

                # Create a temporary dataframe to align indices
                temp_df = pd.DataFrame({'%-true_ofi': ofi, '%-whale_volume': whale_vol})

                # Set dataframe index to merge
                dataframe = dataframe.set_index('date')
                # Join and fill missing with 0
                dataframe = dataframe.join(temp_df).fillna({'%-true_ofi': 0, '%-whale_volume': 0})
                dataframe = dataframe.reset_index()
            else:
                dataframe['%-true_ofi'] = 0
                dataframe['%-whale_volume'] = 0
        else:
            dataframe['%-true_ofi'] = 0
            dataframe['%-whale_volume'] = 0

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
            dataframe["&-s_close"] > self.buy_threshold.value
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "freqai_long")

        enter_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-s_close"] < self.sell_threshold.value
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
