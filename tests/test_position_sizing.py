# tests/test_position_sizing.py
import numpy as np
import pandas as pd
from datetime import datetime
from freqtrade.resolvers import StrategyResolver
from freqtrade.configuration import Configuration

class MockDataProvider:
    def __init__(self, df):
        self.df = df

    def get_analyzed_dataframe(self, pair, timeframe):
        return self.df, datetime.now()

def test_position_sizing_logic():
    # 1. Load StatArbStrategy
    config = Configuration.from_files(["user_data/config.json"])
    config["strategy_path"] = "user_data/strategies"
    config["strategy"] = "StatArbStrategy"
    config["trading_mode"] = "futures"
    config["margin_mode"] = "isolated"
    
    strategy = StrategyResolver.load_strategy(config)
    proposed_stake = 100.0
    min_stake = 10.0
    max_stake = 500.0
    leverage = 1.0
    current_time = datetime.now()
    current_rate = 1.0
    pair = "SOL/USDT:USDT"
    
    # Enable dynamic position sizing
    strategy.use_volatility_staking.value = 1.0
    strategy.base_volatility.value = 0.0015
    strategy.min_stake_multiplier.value = 0.2
    strategy.max_stake_multiplier.value = 1.5

    # Case A: Normal Volatility (identical to base_volatility)
    df_normal = pd.DataFrame({
        "date": [current_time],
        "volatility": [0.0015]
    })
    strategy.dp = MockDataProvider(df_normal)
    stake_normal = strategy.custom_stake_amount(
        pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, "long", "long"
    )
    assert abs(stake_normal - proposed_stake) < 1e-3

    # Case B: High Volatility (5x baseline) -> Should scale down to min_multiplier (0.2x)
    df_high = pd.DataFrame({
        "date": [current_time],
        "volatility": [0.0075] # multiplier = 0.0015 / 0.0075 = 0.2
    })
    strategy.dp = MockDataProvider(df_high)
    stake_high = strategy.custom_stake_amount(
        pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, "long", "long"
    )
    assert abs(stake_high - (proposed_stake * 0.2)) < 1e-3

    # Case C: Low Volatility (0.5x baseline) -> Should scale up to max_multiplier (1.5x) instead of 2x
    df_low = pd.DataFrame({
        "date": [current_time],
        "volatility": [0.00075] # multiplier = 0.0015 / 0.00075 = 2.0 -> clamped to 1.5
    })
    strategy.dp = MockDataProvider(df_low)
    stake_low = strategy.custom_stake_amount(
        pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, "long", "long"
    )
    assert abs(stake_low - (proposed_stake * 1.5)) < 1e-3

    # Case D: Feature Disabled -> Should return proposed_stake exactly
    strategy.use_volatility_staking.value = 0.0
    strategy.dp = MockDataProvider(df_low) # even with low vol
    stake_disabled = strategy.custom_stake_amount(
        pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, "long", "long"
    )
    assert abs(stake_disabled - proposed_stake) < 1e-3

    # Reset feature toggle
    strategy.use_volatility_staking.value = 1.0

    # Case E: Missing Volatility Data -> Should fall back gracefully to proposed_stake
    df_missing = pd.DataFrame({
        "date": [current_time]
        # volatility column is missing
    })
    strategy.dp = MockDataProvider(df_missing)
    stake_missing = strategy.custom_stake_amount(
        pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, "long", "long"
    )
    assert abs(stake_missing - proposed_stake) < 1e-3

    # Case F: Empty Dataframe -> Should fall back gracefully to proposed_stake
    strategy.dp = MockDataProvider(pd.DataFrame())
    stake_empty = strategy.custom_stake_amount(
        pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, "long", "long"
    )
    assert abs(stake_empty - proposed_stake) < 1e-3
