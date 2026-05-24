# tests/test_cointegration.py
import numpy as np
import pandas as pd
from freqtrade.resolvers import StrategyResolver
from freqtrade.configuration import Configuration

class MockDataProvider:
    def __init__(self, dataframes, whitelist):
        self.dataframes = dataframes
        self._whitelist = whitelist

    def current_whitelist(self):
        return self._whitelist

    def get_pair_dataframe(self, pair, timeframe):
        return self.dataframes.get(pair, pd.DataFrame())

def test_cointegration_analysis():
    # 1. Load StatArbStrategy
    config = Configuration.from_files(["user_data/config.json"])
    config["strategy_path"] = "user_data/strategies"
    config["strategy"] = "StatArbStrategy"
    config["trading_mode"] = "futures"
    config["margin_mode"] = "isolated"
    
    strategy = StrategyResolver.load_strategy(config)
    
    # Generate reproducible data using seed
    np.random.seed(42)
    length = 600
    
    # x is the cointegration partner (ETH)
    x = np.cumsum(np.random.normal(0, 0.01, length)) + 5.0
    # eps is stationary white noise
    eps = np.random.normal(0, 0.005, length)
    # y is the self pair (SOL), perfectly cointegrated with x
    beta = 1.5
    alpha = 0.5
    y = beta * x + alpha + eps
    
    # z is the independent pair (BTC)
    z = np.cumsum(np.random.normal(0, 0.01, length)) + 5.0
    
    dates = pd.date_range(start="2026-05-01", periods=length, freq="5min")
    
    # Build DataFrames
    df_self = pd.DataFrame({
        "date": dates,
        "open": np.exp(y),
        "high": np.exp(y),
        "low": np.exp(y),
        "close": np.exp(y),
        "volume": [100.0] * length
    })
    
    df_eth = pd.DataFrame({
        "date": dates,
        "open": np.exp(x),
        "high": np.exp(x),
        "low": np.exp(x),
        "close": np.exp(x),
        "volume": [100.0] * length
    })
    
    df_btc = pd.DataFrame({
        "date": dates,
        "open": np.exp(z),
        "high": np.exp(z),
        "low": np.exp(z),
        "close": np.exp(z),
        "volume": [100.0] * length
    })
    
    # Whitelist and DataProvider injection
    whitelist = ["SOL/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"]
    dataframes = {
        "SOL/USDT:USDT": df_self,
        "ETH/USDT:USDT": df_eth,
        "BTC/USDT:USDT": df_btc
    }
    
    strategy.dp = MockDataProvider(dataframes, whitelist)
    
    # Test internal helper _find_best_cointegration_partner
    log_self = y
    log_others = {
        "ETH/USDT:USDT": x,
        "BTC/USDT:USDT": z
    }
    other_pairs = ["ETH/USDT:USDT", "BTC/USDT:USDT"]
    
    best_op, best_beta, best_alpha, best_pvalue = strategy._find_best_cointegration_partner(
        log_self, log_others, other_pairs, 0, length
    )
    
    # ETH should be the best partner because y is constructed to be cointegrated with x
    assert best_op == "ETH/USDT:USDT"
    # Beta and alpha should be close to original parameters
    assert abs(best_beta - beta) < 0.1
    assert abs(best_alpha - alpha) < 0.1
    # P-value should be extremely small (cointegrated relationship)
    assert best_pvalue < 0.05
    
    # Test calculate_indicators
    metadata = {"pair": "SOL/USDT:USDT"}
    result_df = strategy.calculate_indicators(df_self.copy(), metadata)
    
    # Should have 'spread' and 'coint_active' columns
    assert "spread" in result_df.columns
    assert "coint_active" in result_df.columns
    
    # Under cointegration, the active flag should be 1
    assert result_df["coint_active"].iloc[-1] == 1
    
    # 2. Test scenario where no cointegrating relationship exists
    # If self is independent of both ETH and BTC
    np.random.seed(2)
    y_indep = np.cumsum(np.random.normal(0, 0.01, length)) + 5.0
    df_self_indep = pd.DataFrame({
        "date": dates,
        "open": np.exp(y_indep),
        "high": np.exp(y_indep),
        "low": np.exp(y_indep),
        "close": np.exp(y_indep),
        "volume": [100.0] * length
    })
    
    dataframes["SOL/USDT:USDT"] = df_self_indep
    
    best_op_indep, _, _, best_pvalue_indep = strategy._find_best_cointegration_partner(
        y_indep, log_others, other_pairs, 0, length
    )
    
    # Under independence, the active flag should resolve to 0
    result_df_indep = strategy.calculate_indicators(df_self_indep.copy(), metadata)
    assert "coint_active" in result_df_indep.columns
    assert result_df_indep["coint_active"].iloc[-1] == 0
