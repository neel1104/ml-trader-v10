# user_data/strategies/StatArbStrategy.py
import pandas as pd
import numpy as np
from functools import reduce
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import talib.abstract as ta
import logging

logger = logging.getLogger(__name__)

class StatArbStrategy(IStrategy):
    """
    Phase 2: Market-Neutral Execution - Cointegration-Based Statistical Arbitrage
    Enhanced with FreqAI for residual return modeling.
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

    # Fee settings
    use_custom_fee = True

    # Hyperopt parameters
    zscore_entry = DecimalParameter(2.0, 4.0, default=2.5, space='buy')
    zscore_exit = DecimalParameter(-0.5, 0.5, default=0, space='sell')
    lookback_period = IntParameter(20, 200, default=100, space='buy')
    
    # Target specific magnitudes
    min_predicted_magnitude = DecimalParameter(0.5, 2.0, default=1.0, space='buy')
    min_reversion_speed = DecimalParameter(0.2, 1.0, default=0.5, space='buy')
    
    # Toggles to let hyperopt decide which filters are actually helping
    use_freqai_reversion = DecimalParameter(0, 1, default=0.5, space='buy') # > 0.5 means use it
    use_rsi_filter = DecimalParameter(0, 1, default=0.5, space='buy')
    use_mfi_filter = DecimalParameter(0, 1, default=0.5, space='buy')
    use_ofi_filter = DecimalParameter(0, 1, default=0.5, space='buy')

    rsi_threshold_long = DecimalParameter(30, 50, default=45, space='buy')
    rsi_threshold_short = DecimalParameter(50, 70, default=55, space='buy')
    mfi_threshold_long = DecimalParameter(30, 50, default=45, space='buy')
    mfi_threshold_short = DecimalParameter(50, 70, default=55, space='buy')

    def informative_pairs(self):
        """
        Define the pairs to be loaded synchronously.
        We always need BTC/USDT to calculate relative spreads for other pairs.
        """
        return [
            ("BTC/USDT:USDT", self.timeframe),
            ("ETH/USDT:USDT", self.timeframe),
            ("SOL/USDT:USDT", self.timeframe),
            ("AVAX/USDT:USDT", self.timeframe),
        ]

    def calculate_zscore(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair")
        # If the pair is BTC, we spread against ETH. Otherwise, spread against BTC.
        other_pair = "ETH/USDT:USDT" if pair == "BTC/USDT:USDT" else "BTC/USDT:USDT"
        other_df = self.dp.get_pair_dataframe(other_pair, self.timeframe)
        
        if not other_df.empty:
            # Align dataframes
            temp_df = pd.merge(dataframe[['date', 'close']], other_df[['date', 'close']], on='date', suffixes=('', '_other'), how='left')
            temp_df['close_other'] = temp_df['close_other'].ffill()
            
            # Spread = log(PriceA) - log(PriceB)
            spread = np.log(temp_df['close']) - np.log(temp_df['close_other'])
            
            # Rolling Z-score of the spread
            lookback = self.lookback_period.value
            spread_mean = spread.rolling(window=lookback).mean()
            spread_std = spread.rolling(window=lookback).std()
            dataframe['%-zscore'] = (spread - spread_mean) / (spread_std + 0.0001)
            dataframe['%-zscore_diff'] = dataframe['%-zscore'].diff()
            dataframe['%-volatility'] = spread.rolling(window=30).std()
            
            # print(f"DEBUG: {pair} spread min/max: {spread.min():.4f}/{spread.max():.4f}, zscore min/max: {dataframe['%-zscore'].min():.4f}/{dataframe['%-zscore'].max():.4f}")
        else:
            # print(f"DEBUG: Empty data for {other_pair} while processing {pair}")
            dataframe['%-zscore'] = 0
            dataframe['%-zscore_diff'] = 0
            dataframe['%-volatility'] = 0
            
        return dataframe

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, **kwargs) -> pd.DataFrame:
        """
        Feature Engineering (Phase 1 & 3)
        """
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Stat-Arb specific features (Phase 2)
        """
        metadata = kwargs.get("metadata")
        dataframe = self.calculate_zscore(dataframe, metadata)
        
        # Microstructure integration
        if 'taker_buy_base_volume' in dataframe.columns:
             dataframe['%-ofi'] = (2 * dataframe['taker_buy_base_volume']) - dataframe['volume']
        else:
             dataframe['%-ofi'] = 0

        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Target: predict the Z-score mean reversion (3 candles into the future)
        metadata = kwargs.get("metadata")
        dataframe = self.calculate_zscore(dataframe, metadata)
        dataframe["&-zscore_target"] = dataframe["%-zscore"].shift(-3)
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe = self.calculate_zscore(dataframe, metadata)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Long Entry: Z-score low (Mandatory)
        enter_long_conditions = [
            dataframe["%-zscore"] < -self.zscore_entry.value,
            dataframe["%-zscore_diff"] > self.min_reversion_speed.value # Speed check
        ]
        
        # Optional Stacking
        if self.use_freqai_reversion.value > 0.5:
             enter_long_conditions.append(dataframe["do_predict"] == 1)
             enter_long_conditions.append(dataframe["&-zscore_target"] > dataframe["%-zscore"])
             # Predicted reversion magnitude must be large enough
             enter_long_conditions.append((dataframe["&-zscore_target"] - dataframe["%-zscore"]) > self.min_predicted_magnitude.value)
        
        if self.use_rsi_filter.value > 0.5:
             enter_long_conditions.append(dataframe.get("%-rsi-14", dataframe.get("%-rsi-14_5m", 50)) < self.rsi_threshold_long.value)
             
        if self.use_mfi_filter.value > 0.5:
             enter_long_conditions.append(dataframe.get("%-mfi-14", dataframe.get("%-mfi-14_5m", 50)) < self.mfi_threshold_long.value)
             
        if self.use_ofi_filter.value > 0.5:
             enter_long_conditions.append(dataframe.get("%-ofi", dataframe.get("%-ofi_5m", 0)) > 0)

        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "stat_arb_long_stacked")

        # Short Entry: Z-score high (Mandatory)
        enter_short_conditions = [
            dataframe["%-zscore"] > self.zscore_entry.value,
            dataframe["%-zscore_diff"] < -self.min_reversion_speed.value
        ]
        
        if self.use_freqai_reversion.value > 0.5:
             enter_short_conditions.append(dataframe["do_predict"] == 1)
             enter_short_conditions.append(dataframe["&-zscore_target"] < dataframe["%-zscore"])
             enter_short_conditions.append((dataframe["%-zscore"] - dataframe["&-zscore_target"]) > self.min_predicted_magnitude.value)
             
        if self.use_rsi_filter.value > 0.5:
             enter_short_conditions.append(dataframe.get("%-rsi-14", dataframe.get("%-rsi-14_5m", 50)) > self.rsi_threshold_short.value)
             
        if self.use_mfi_filter.value > 0.5:
             enter_short_conditions.append(dataframe.get("%-mfi-14", dataframe.get("%-mfi-14_5m", 50)) > self.mfi_threshold_short.value)
             
        if self.use_ofi_filter.value > 0.5:
             enter_short_conditions.append(dataframe.get("%-ofi", dataframe.get("%-ofi_5m", 0)) < 0)

        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "stat_arb_short_stacked")

        return dataframe


    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit when Z-score returns to mean
        exit_long_conditions = [
            # dataframe["do_predict"] == 1,
            dataframe["%-zscore"] >= self.zscore_exit.value
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            # dataframe["do_predict"] == 1,
            dataframe["%-zscore"] <= -self.zscore_exit.value
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return dataframe

    def custom_fee(self, pair: str, trade_type: str, temporary_entry_fee: float, temporary_exit_fee: float,
                   actual_fee: float, **kwargs) -> float:
        """
        Force maker fee (0.02%) for backtesting Maker-Only execution.
        """
        return 0.0002
