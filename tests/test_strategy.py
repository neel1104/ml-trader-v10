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
