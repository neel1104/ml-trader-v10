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

def test_hyperopt_parameters():
    from freqtrade.configuration import Configuration
    from freqtrade.resolvers import StrategyResolver
    config = Configuration.from_files(["user_data/config.json"])
    config["strategy_path"] = "user_data/strategies"
    config["strategy"] = "CatBoostStrategy"
    config["trading_mode"] = "futures"
    config["margin_mode"] = "isolated"
    strategy = StrategyResolver.load_strategy(config)
    
    assert hasattr(strategy.__class__, 'buy_threshold')
    assert hasattr(strategy.__class__, 'sell_threshold')

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

def test_smart_money_features():
    from freqtrade.configuration import Configuration
    from freqtrade.resolvers import StrategyResolver
    config = Configuration.from_files(["user_data/config.json"])
    config["strategy_path"] = "user_data/strategies"
    config["strategy"] = "CatBoostStrategy"
    config["trading_mode"] = "futures"
    config["margin_mode"] = "isolated"
    strategy = StrategyResolver.load_strategy(config)
    
    import pandas as pd
    import numpy as np
    
    # Mock OHLCV with Binance taker volume columns
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-04-26 12:00", "2026-04-26 12:05"]),
        "open": [100, 100],
        "close": [101, 99], 
        "high": [102, 103], 
        "low": [99, 98], 
        "volume": [100, 100],
        "taker_buy_base_volume": [65, 30] # 65% buy in candle 1, 30% buy in candle 2
    })
    
    result = strategy.feature_engineering_expand_all(df, period=14, metadata={'pair': 'BTC/USDT:USDT'})
    
    assert "%-true_ofi" in result.columns
    assert "%-whale_volume" in result.columns
    # Check OFI calculation: (2 * 65) - 100 = 30
    assert result.iloc[0]["%-true_ofi"] == 30
    # Check OFI calculation: (2 * 30) - 100 = -40
    assert result.iloc[1]["%-true_ofi"] == -40
    # Check Whale Proxy (65% > 60% and 30% < 40% should trigger)
    assert result.iloc[0]["%-whale_volume"] == 100
    assert result.iloc[1]["%-whale_volume"] == 100
