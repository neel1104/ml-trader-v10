import pandas as pd
import numpy as np
from functools import reduce
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, merge_informative_pair
import talib.abstract as ta
import logging
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress

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

    # Cointegration parameters (fixed to safe stable defaults)
    coint_window = IntParameter(200, 1000, default=500, space='buy', optimize=False)
    coint_pvalue_threshold = DecimalParameter(0.01, 0.20, default=0.10, space='buy', optimize=False)

    # Dynamic position sizing parameters (fixed to stable defaults, hyperopt-ready)
    use_volatility_staking = DecimalParameter(0, 1, default=1.0, space='buy', optimize=False)
    base_volatility = DecimalParameter(0.0001, 0.01, default=0.0015, decimals=4, space='buy', optimize=False)
    min_stake_multiplier = DecimalParameter(0.05, 0.5, default=0.2, space='buy', optimize=False)
    max_stake_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space='buy', optimize=False)

    # Slippage adjustment (simulated in custom_fee)
    # Default is 0.0005 (0.05% or 5 bps per leg, round-trip 10 bps friction)
    backtest_slippage = DecimalParameter(0.0, 0.0020, default=0.0005, decimals=4, space='buy', optimize=False)

    # Guardrails (fixed to stable defaults to keep hyperopt search space small and prevent zero-trade premature convergence)
    use_cvd_filter = DecimalParameter(0, 1, default=0.0, space='buy', optimize=False)
    use_ofi_filter = DecimalParameter(0, 1, default=0.0, space='buy', optimize=False)
    use_rsi_filter = DecimalParameter(0, 1, default=1.0, space='buy', optimize=False)
    use_mfi_filter = DecimalParameter(0, 1, default=0.0, space='buy', optimize=False)
    use_trend_filter = DecimalParameter(0, 1, default=1.0, space='buy', optimize=False)
    use_freqai_reversion = DecimalParameter(0, 1, default=1.0, space='buy', optimize=False)

    rsi_threshold_long = DecimalParameter(20, 50, default=30, space='buy', optimize=False)
    rsi_threshold_short = DecimalParameter(50, 80, default=70, space='buy', optimize=False)
    mfi_threshold_long = DecimalParameter(10, 40, default=20, space='buy', optimize=False)
    mfi_threshold_short = DecimalParameter(60, 90, default=80, space='buy', optimize=False)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        informative_pairs.append(("BTC/USDT:USDT", "1h"))
        return informative_pairs

    def _find_best_cointegration_partner(self, log_self, log_others, other_pairs, start_idx, end_idx):
        best_op = None
        best_adf = float('inf')
        best_beta = 1.0
        best_alpha = 0.0
        best_pvalue = 1.0
        
        y_train = log_self[start_idx:end_idx]
        
        for op in other_pairs:
            if op not in log_others:
                continue
            x_train = log_others[op][start_idx:end_idx]
            
            try:
                beta, alpha, _, _, _ = linregress(x_train, y_train)
                if np.isnan(beta) or np.isinf(beta):
                    continue
                residuals = y_train - (beta * x_train + alpha)
                adf_res = adfuller(residuals)
                adf_stat = adf_res[0]
                pval = adf_res[1]
                
                if adf_stat < best_adf:
                    best_adf = adf_stat
                    best_pvalue = pval
                    best_beta = beta
                    best_alpha = alpha
                    best_op = op
            except Exception:
                continue
                
        return best_op, best_beta, best_alpha, best_pvalue

    def _apply_cointegration_params(self, spread, coint_active, log_self, log_others, start_idx, end_idx, best_op, best_beta, best_alpha, best_pvalue, pair):
        if best_op is not None:
            spread[start_idx:end_idx] = log_self[start_idx:end_idx] - (best_beta * log_others[best_op][start_idx:end_idx] + best_alpha)
            coint_active[start_idx:end_idx] = 1 if best_pvalue <= self.coint_pvalue_threshold.value else 0
        else:
            fallback_op = "BTC/USDT:USDT" if pair != "BTC/USDT:USDT" else "ETH/USDT:USDT"
            if fallback_op in log_others:
                spread[start_idx:end_idx] = log_self[start_idx:end_idx] - log_others[fallback_op][start_idx:end_idx]
            else:
                spread[start_idx:end_idx] = 0.0
            coint_active[start_idx:end_idx] = 0

    def calculate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculates all technical indicators used by the strategy.
        These are NOT prefixed with %- to ensure they are available
        during backtesting/hyperopt even if FreqAI cleans up features.
        """
        pair = metadata.get("pair")
        whitelist = self.dp.current_whitelist()
        other_pairs = [p for p in whitelist if p != pair]
        
        # Load and align all other pairs dataframes
        aligned_df = dataframe[['date', 'close']].copy()
        aligned_df = aligned_df.rename(columns={'close': 'close_self'})
        
        for op in other_pairs:
            op_df = self.dp.get_pair_dataframe(op, self.timeframe)
            if not op_df.empty:
                op_df_subset = op_df[['date', 'close']].rename(columns={'close': f'close_{op}'})
                aligned_df = pd.merge(aligned_df, op_df_subset, on='date', how='left')
                aligned_df[f'close_{op}'] = aligned_df[f'close_{op}'].ffill().bfill()
        
        n_candles = len(dataframe)
        spread = np.zeros(n_candles)
        coint_active = np.zeros(n_candles, dtype=int)
        
        coint_win = self.coint_window.value
        step = 288
        
        log_self = np.log(aligned_df['close_self'].to_numpy())
        log_others = {op: np.log(aligned_df[f'close_{op}'].to_numpy()) for op in other_pairs if f'close_{op}' in aligned_df.columns}
        
        # 1. First block: [0, min(coint_win, n_candles)] (Warm-up block, coint_active strictly 0 to prevent look-ahead bias)
        first_block_end = min(coint_win, n_candles)
        fallback_op = "BTC/USDT:USDT" if pair != "BTC/USDT:USDT" else "ETH/USDT:USDT"
        if fallback_op in log_others:
            spread[0:first_block_end] = log_self[0:first_block_end] - log_others[fallback_op][0:first_block_end]
        else:
            spread[0:first_block_end] = 0.0
        coint_active[0:first_block_end] = 0
            
        # 2. Subsequent blocks: i from coint_win to n_candles by step (288)
        if n_candles > coint_win:
            for i in range(coint_win, n_candles, step):
                block_start = i
                block_end = min(i + step, n_candles)
                
                # Window for regression is [i - coint_win, i]
                win_start = i - coint_win
                win_end = i
                
                best_op, best_beta, best_alpha, best_pvalue = self._find_best_cointegration_partner(
                    log_self, log_others, other_pairs, win_start, win_end
                )
                self._apply_cointegration_params(
                    spread, coint_active, log_self, log_others, block_start, block_end,
                    best_op, best_beta, best_alpha, best_pvalue, pair
                )
                
        dataframe['spread'] = spread
        dataframe['coint_active'] = coint_active
        
        # Calculate rolling z-score using spread
        lookback = self.lookback_period.value
        spread_mean = dataframe['spread'].rolling(window=lookback).mean()
        spread_std = dataframe['spread'].rolling(window=lookback).std()
        
        dataframe['zscore'] = ((dataframe['spread'] - spread_mean) / (spread_std + 0.0001)).fillna(0)
        dataframe['zscore_diff'] = dataframe['zscore'].diff().fillna(0)
        dataframe['volatility'] = dataframe['spread'].rolling(window=30).std().fillna(0)

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
            dataframe["coint_active"] == 1,
            (fav_long & (dataframe["zscore"] < -self.zscore_fav.value)) |
            (neu_long & (dataframe["zscore"] < -self.zscore_neu.value)) |
            (pen_long & (dataframe["zscore"] < -self.zscore_pen.value)),
            dataframe["do_predict"] == 1,
            dataframe["&-zscore_target"] > dataframe["zscore"],
            dataframe["expected_reversion"] > self.min_expected_profit.value
        ]

        # Check if microstructure data actually exists in this dataset
        has_microstructure = ("%-ofi" in dataframe.columns) and (dataframe["%-ofi"].any())

        if self.use_trend_filter.value > 0.5:
             enter_long_conditions.append(dataframe["btc_above_ema_1h"] == 1)
        if self.use_rsi_filter.value > 0.5:
             enter_long_conditions.append(dataframe.get("%-rsi-14", dataframe.get("%-rsi-14_5m", 50)) < self.rsi_threshold_long.value)
        if has_microstructure:
             if self.use_ofi_filter.value > 0.5:
                  enter_long_conditions.append(dataframe.get("%-ofi", dataframe.get("%-ofi_5m", 0)) > 0)
             if self.use_cvd_filter.value > 0.5:
                  enter_long_conditions.append(dataframe.get("%-cvd_zscore_1h", dataframe.get("%-cvd_zscore_1h_5m", 0)) > 0)

        dataframe.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "stat_arb_long")

        fav_short = (dataframe['funding_benefit_short'] > 0.0001)
        pen_short = (dataframe['funding_benefit_short'] < -0.0001)
        neu_short = ~(fav_short | pen_short)

        enter_short_conditions = [
            dataframe["coint_active"] == 1,
            (fav_short & (dataframe["zscore"] > self.zscore_fav.value)) |
            (neu_short & (dataframe["zscore"] > self.zscore_neu.value)) |
            (pen_short & (dataframe["zscore"] > self.zscore_pen.value)),
            dataframe["do_predict"] == 1,
            dataframe["&-zscore_target"] < dataframe["zscore"],
            dataframe["expected_reversion"] > self.min_expected_profit.value
        ]

        if self.use_trend_filter.value > 0.5:
             enter_short_conditions.append(dataframe["btc_above_ema_1h"] == 0)
        if has_microstructure:
             if self.use_ofi_filter.value > 0.5:
                  enter_short_conditions.append(dataframe.get("%-ofi", dataframe.get("%-ofi_5m", 0)) < 0)
             if self.use_cvd_filter.value > 0.5:
                  enter_short_conditions.append(dataframe.get("%-cvd_zscore_1h", dataframe.get("%-cvd_zscore_1h_5m", 0)) < 0)

        dataframe.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "stat_arb_short")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["zscore"] >= self.zscore_exit.value, "exit_long"] = 1
        dataframe.loc[dataframe["zscore"] <= -self.zscore_exit.value, "exit_short"] = 1
        return dataframe

    def custom_fee(self, pair: str, trade_type: str, temporary_entry_fee: float, temporary_exit_fee: float, actual_fee: float, **kwargs) -> float:
        # Base exchange maker fee is 0.02% (0.0002) + backtest slippage friction
        return 0.0002 + float(self.backtest_slippage.value)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        """
        Dynamically adjust the stake size based on the rolling spread volatility.
        """
        if self.use_volatility_staking.value < 0.5:
            return proposed_stake

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return proposed_stake
            
            volatility = dataframe.iloc[-1].get("volatility", 0.0)
            if pd.isna(volatility) or volatility <= 0.0:
                return proposed_stake
            
            # Scale dynamically using the empirical baseline target volatility
            multiplier = self.base_volatility.value / (volatility + 1e-8)
            
            # Clamp multiplier to boundaries
            min_mult = self.min_stake_multiplier.value
            max_mult = self.max_stake_multiplier.value
            multiplier = max(min_mult, min(max_mult, multiplier))
            
            # Calculate dynamic stake size
            dynamic_stake = proposed_stake * multiplier
            
            # Bounding limits
            if min_stake is not None:
                dynamic_stake = max(min_stake, dynamic_stake)
            if max_stake is not None:
                dynamic_stake = min(max_stake, dynamic_stake)
                
            return float(dynamic_stake)
            
        except Exception as e:
            logger.error(f"Error in custom_stake_amount for {pair}: {e}")
            return proposed_stake
