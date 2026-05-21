import pandas as pd
import numpy as np
from functools import reduce
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, merge_informative_pair
import talib.abstract as ta
import logging

logger = logging.getLogger(__name__)

class StatArbStrategy(IStrategy):
    """
    Phase 2: Market-Neutral Execution - Cointegration-Based Statistical Arbitrage
    Enhanced with FreqAI for residual return modeling.
    Deterministic stable version.
    """
    timeframe = "5m"
    can_short = True
    startup_candle_count = 100

    # Mandatory attributes
    stoploss = -0.05
    minimal_roi = {
        "0": 0.03
    }
    
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Hyperoptable parameters
    lookback_period = IntParameter(20, 300, default=100, space='buy')
    zscore_fav = DecimalParameter(1.0, 3.0, default=1.5, space='buy')
    zscore_neu = DecimalParameter(1.5, 3.5, default=2.0, space='buy')
    zscore_pen = DecimalParameter(2.0, 4.5, default=3.0, space='buy')
    zscore_exit = DecimalParameter(-1.0, 1.0, default=0.0, space='sell')
    
    min_expected_profit = DecimalParameter(0.0001, 0.01, default=0.001, space='buy')
    min_reversion_speed = DecimalParameter(0.01, 1.0, default=0.1, space='buy')

    # Guardrails
    use_cvd_filter = DecimalParameter(0, 1, default=0.5, space='buy')
    use_ofi_filter = DecimalParameter(0, 1, default=0.5, space='buy')
    use_rsi_filter = DecimalParameter(0, 1, default=0.5, space='buy')
    use_mfi_filter = DecimalParameter(0, 1, default=0.5, space='buy')
    use_trend_filter = DecimalParameter(0, 1, default=0.5, space='buy')
    use_freqai_reversion = DecimalParameter(0, 1, default=1.0, space='buy')

    rsi_threshold_long = DecimalParameter(20, 50, default=30, space='buy')
    rsi_threshold_short = DecimalParameter(50, 80, default=70, space='buy')
    mfi_threshold_long = DecimalParameter(10, 40, default=20, space='buy')
    mfi_threshold_short = DecimalParameter(60, 90, default=80, space='buy')

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        informative_pairs.append(("BTC/USDT:USDT", "1h"))
        return informative_pairs

    def calculate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculates all technical indicators used by the strategy.
        These are NOT prefixed with %- to ensure they are available
        during backtesting/hyperopt even if FreqAI cleans up features.
        """
        pair = metadata.get("pair")
        other_pair = "BTC/USDT:USDT"
        if pair == other_pair:
             other_pair = "ETH/USDT:USDT"

        other_df = self.dp.get_pair_dataframe(other_pair, self.timeframe)
        
        if not other_df.empty:
            temp_df = pd.merge(dataframe[['date', 'close']], other_df[['date', 'close']], on='date', suffixes=('', '_other'), how='left')
            temp_df['close_other'] = temp_df['close_other'].ffill().bfill()
            spread = np.log(temp_df['close']) - np.log(temp_df['close_other'])
            
            lookback = self.lookback_period.value
            spread_mean = spread.rolling(window=lookback).mean()
            spread_std = spread.rolling(window=lookback).std()
            
            dataframe['zscore'] = ((spread - spread_mean) / (spread_std + 0.0001)).fillna(0)
            dataframe['zscore_diff'] = dataframe['zscore'].diff().fillna(0)
            dataframe['volatility'] = spread.rolling(window=30).std().fillna(0)
        else:
            dataframe['zscore'] = 0.0
            dataframe['zscore_diff'] = 0.0
            dataframe['volatility'] = 0.0

        # Funding indicators
        if 'funding_rate' in dataframe.columns:
            dataframe['funding_benefit_long'] = -dataframe['funding_rate'].fillna(0)
            dataframe['funding_benefit_short'] = dataframe['funding_rate'].fillna(0)
        else:
            dataframe['funding_benefit_long'] = 0.0
            dataframe['funding_benefit_short'] = 0.0
            
        return dataframe

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        metadata = kwargs.get("metadata")
        dataframe = self.calculate_indicators(dataframe, metadata)

        # Map to FreqAI features
        dataframe['%-zscore'] = dataframe['zscore']
        dataframe['%-zscore_diff'] = dataframe['zscore_diff']
        dataframe['%-volatility'] = dataframe['volatility']
        dataframe['%-funding_benefit_long'] = dataframe['funding_benefit_long']
        dataframe['%-funding_benefit_short'] = dataframe['funding_benefit_short']

        if 'taker_buy_base_volume' in dataframe.columns:
             taker_sell_base_volume = (dataframe['volume'] - dataframe['taker_buy_base_volume']).fillna(0)
             volume_delta = (dataframe['taker_buy_base_volume'] - taker_sell_base_volume).fillna(0)
             dataframe['%-ofi'] = volume_delta
             cvd_1h = volume_delta.rolling(window=12).sum().fillna(0)
             cvd_4h = volume_delta.rolling(window=48).sum().fillna(0)
             dataframe['%-cvd_zscore_1h'] = ((cvd_1h - cvd_1h.rolling(window=288).mean()) / (cvd_1h.rolling(window=288).std() + 0.0001)).fillna(0)
             dataframe['%-cvd_zscore_4h'] = ((cvd_4h - cvd_4h.rolling(window=288).mean()) / (cvd_4h.rolling(window=288).std() + 0.0001)).fillna(0)
        else:
             dataframe['%-ofi'] = 0.0
             dataframe['%-cvd_zscore_1h'] = 0.0
             dataframe['%-cvd_zscore_4h'] = 0.0

        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        metadata = kwargs.get("metadata")
        dataframe = self.calculate_indicators(dataframe, metadata)
        dataframe["&-zscore_target"] = dataframe["zscore"].shift(-3).fillna(0)
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        btc_1h = self.dp.get_pair_dataframe("BTC/USDT:USDT", "1h")
        if not btc_1h.empty:
            btc_1h['ema_200'] = ta.EMA(btc_1h, timeperiod=200)
            dataframe = merge_informative_pair(dataframe, btc_1h, self.timeframe, "1h", ffill=True)
            dataframe['btc_above_ema_1h'] = (dataframe['close_1h'] > dataframe['ema_200_1h']).astype(int)
        else:
            dataframe['btc_above_ema_1h'] = 1

        dataframe = self.calculate_indicators(dataframe, metadata)
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['expected_reversion'] = ((dataframe['&-zscore_target'] - dataframe['zscore']).abs() * dataframe['volatility'])

        fav_long = (dataframe['funding_benefit_long'] > 0.0001)
        pen_long = (dataframe['funding_benefit_long'] < -0.0001)
        neu_long = ~(fav_long | pen_long)

        enter_long_conditions = [
            (fav_long & (dataframe["zscore"] < -self.zscore_fav.value)) |
            (neu_long & (dataframe["zscore"] < -self.zscore_neu.value)) |
            (pen_long & (dataframe["zscore"] < -self.zscore_pen.value)),
            dataframe["do_predict"] == 1,
            dataframe["&-zscore_target"] > dataframe["zscore"],
            dataframe["expected_reversion"] > self.min_expected_profit.value
        ]

        if self.use_trend_filter:
             enter_long_conditions.append(dataframe["btc_above_ema_1h"] == 1)
        if self.use_rsi_filter:
             enter_long_conditions.append(dataframe.get("%-rsi-14", dataframe.get("%-rsi-14_5m", 50)) < self.rsi_threshold_long.value)
        if self.use_ofi_filter:
             enter_long_conditions.append(dataframe.get("%-ofi", dataframe.get("%-ofi_5m", 0)) > 0)
        if self.use_cvd_filter:
             enter_long_conditions.append(dataframe.get("%-cvd_zscore_1h", dataframe.get("%-cvd_zscore_1h_5m", 0)) > 0)

        dataframe.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "stat_arb_long")

        fav_short = (dataframe['funding_benefit_short'] > 0.0001)
        pen_short = (dataframe['funding_benefit_short'] < -0.0001)
        neu_short = ~(fav_short | pen_short)

        enter_short_conditions = [
            (fav_short & (dataframe["zscore"] > self.zscore_fav.value)) |
            (neu_short & (dataframe["zscore"] > self.zscore_neu.value)) |
            (pen_short & (dataframe["zscore"] > self.zscore_pen.value)),
            dataframe["do_predict"] == 1,
            dataframe["&-zscore_target"] < dataframe["zscore"],
            dataframe["expected_reversion"] > self.min_expected_profit.value
        ]

        if self.use_trend_filter:
             enter_short_conditions.append(dataframe["btc_above_ema_1h"] == 0)
        if self.use_ofi_filter:
             enter_short_conditions.append(dataframe.get("%-ofi", dataframe.get("%-ofi_5m", 0)) < 0)
        if self.use_cvd_filter:
             enter_short_conditions.append(dataframe.get("%-cvd_zscore_1h", dataframe.get("%-cvd_zscore_1h_5m", 0)) < 0)

        dataframe.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "stat_arb_short")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["zscore"] >= self.zscore_exit.value, "exit_long"] = 1
        dataframe.loc[dataframe["zscore"] <= -self.zscore_exit.value, "exit_short"] = 1
        return dataframe

    def custom_fee(self, pair: str, trade_type: str, temporary_entry_fee: float, temporary_exit_fee: float, actual_fee: float, **kwargs) -> float:
        return 0.0002
