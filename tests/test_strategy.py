# tests/test_strategy.py
import os
from freqtrade.resolvers import StrategyResolver
from freqtrade.configuration import Configuration

def test_strategy_loads():
    config = Configuration.from_files(["user_data/config.json"])
    # Mocking strategy directory
    config["strategy_path"] = "user_data/strategies"
    config["strategy"] = "CatBoostStrategy"
    config["trading_mode"] = "futures"
    config["margin_mode"] = "isolated"
    
    strategy = StrategyResolver.load_strategy(config)
    assert strategy.timeframe == "5m"
    assert hasattr(strategy, "feature_engineering_expand_all")

def test_microstructure_features():
    from freqtrade.configuration import Configuration
    from freqtrade.resolvers import StrategyResolver
    config = Configuration.from_files(["user_data/config.json"])
    config["strategy_path"] = "user_data/strategies"
    config["strategy"] = "CatBoostStrategy"
    config["trading_mode"] = "futures"
    config["margin_mode"] = "isolated"
    strategy = StrategyResolver.load_strategy(config)
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-04-26 12:00", "2026-04-26 12:05"]),
        "open": [99, 100],
        "close": [100, 101], 
        "high": [102, 103], 
        "low": [99, 100], 
        "volume": [10, 15]
    })
    result = strategy.feature_engineering_expand_basic(df)
    assert "%-spread" in result.columns
    assert "%-mock-cvd" in result.columns
