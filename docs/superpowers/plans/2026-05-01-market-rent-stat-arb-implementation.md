# Market Rent Stat-Arb Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `StatArbStrategy` into a high-frequency, yield-harvesting system using funding regimes and a 22-asset blue-chip universe.

**Architecture:** Use a hub-and-spoke model with BTC as the universal anchor. Net funding benefits are mapped to three discrete entry regimes (Favorable, Neutral, Penalty) to protect principal and harvest "market rent."

**Tech Stack:** Freqtrade, FreqAI (CatBoost), TALib, Pandas.

---

### Task 1: Update Pairlist and Download Data

**Files:**
- Modify: `user_data/config.json`

- [ ] **Step 1: Expand pairlist in config**

Update the `pair_whitelist` to include 22 blue-chip assets.

```json
        "pair_whitelist": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
            "AVAX/USDT:USDT",
            "MATIC/USDT:USDT",
            "LINK/USDT:USDT",
            "ADA/USDT:USDT",
            "DOT/USDT:USDT",
            "ATOM/USDT:USDT",
            "NEAR/USDT:USDT",
            "LTC/USDT:USDT",
            "BCH/USDT:USDT",
            "XRP/USDT:USDT",
            "TRX/USDT:USDT",
            "UNI/USDT:USDT",
            "APT/USDT:USDT",
            "ARB/USDT:USDT",
            "OP/USDT:USDT",
            "SUI/USDT:USDT",
            "INJ/USDT:USDT",
            "RNDR/USDT:USDT",
            "FET/USDT:USDT"
        ]
```

- [ ] **Step 2: Download data for all pairs**

Run the download command to ensure we have funding rates and 5m/1h OHLCV data for the last 60 days.

Run: `freqtrade download-data --config user_data/config.json --days 60 -t 5m 1h --dl-trades`

- [ ] **Step 3: Commit**

```bash
git add user_data/config.json
git commit -m "config: expand pairlist to 22 blue-chip assets"
```

---

### Task 2: Implement Funding Regime Logic

**Files:**
- Modify: `user_data/strategies/StatArbStrategy.py`

- [ ] **Step 1: Update informative_pairs**

Ensure all 22 pairs are handled and BTC is always loaded as the anchor.

```python
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        informative_pairs.append(("BTC/USDT:USDT", "1h"))
        return informative_pairs
```

- [ ] **Step 2: Implement regime detection logic**

Modify `feature_engineering_expand_basic` to calculate the net funding benefit.

```python
    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        metadata = kwargs.get("metadata")
        dataframe = self.calculate_zscore(dataframe, metadata)
        
        # Funding Regime Detection
        if 'funding_rate' in dataframe.columns:
            # Net funding benefit: 
            # If Long: Benefit = -funding_rate (we receive funding if rate is negative)
            # If Short: Benefit = funding_rate (we receive funding if rate is positive)
            dataframe['%-funding_benefit_long'] = -dataframe['funding_rate']
            dataframe['%-funding_benefit_short'] = dataframe['funding_rate']
        else:
            dataframe['%-funding_benefit_long'] = 0
            dataframe['%-funding_benefit_short'] = 0
            
        return dataframe
```

- [ ] **Step 3: Update entry trend with regime thresholds**

Modify `populate_entry_trend` to use discrete Z-score thresholds based on funding.

```python
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Define base thresholds
        z_fav = 2.0
        z_neu = 2.5
        z_pen = 3.5

        # Long Entry: Z-score low
        # Identify Regimes
        fav_long = (dataframe['%-funding_benefit_long'] > 0.0001) # > 0.01% benefit
        pen_long = (dataframe['%-funding_benefit_long'] < -0.0001) # > 0.01% cost
        neu_long = ~(fav_long | pen_long)

        enter_long_conditions = [
            (fav_long & (dataframe["%-zscore"] < -z_fav)) |
            (neu_long & (dataframe["%-zscore"] < -z_neu)) |
            (pen_long & (dataframe["%-zscore"] < -z_pen)),
            dataframe["do_predict"] == 1,
            dataframe["&-zscore_target"] > dataframe["%-zscore"],
            (dataframe["&-zscore_target"] - dataframe["%-zscore"]) > self.min_predicted_magnitude.value
        ]
        
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "stat_arb_long_harvest")

        # Short Entry: Z-score high
        fav_short = (dataframe['%-funding_benefit_short'] > 0.0001)
        pen_short = (dataframe['%-funding_benefit_short'] < -0.0001)
        neu_short = ~(fav_short | pen_short)

        enter_short_conditions = [
            (fav_short & (dataframe["%-zscore"] > z_fav)) |
            (neu_short & (dataframe["%-zscore"] > z_neu)) |
            (pen_short & (dataframe["%-zscore"] > z_pen)),
            dataframe["do_predict"] == 1,
            dataframe["&-zscore_target"] < dataframe["%-zscore"],
            (dataframe["%-zscore"] - dataframe["&-zscore_target"]) > self.min_predicted_magnitude.value
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "stat_arb_short_harvest")

        return dataframe
```

- [ ] **Step 4: Commit**

```bash
git add user_data/strategies/StatArbStrategy.py
git commit -m "strat: implement regime-based funding thresholds"
```

---

### Task 3: FreqAI Optimization for Market Rent

**Files:**
- Modify: `user_data/config.json`

- [ ] **Step 1: Update FreqAI parameters**

Change the identifier to `market_rent_v1` and increase training robustness.

```json
    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 30,
        "backtest_period_days": 7,
        "identifier": "market_rent_v1",
        "model_name": "CatboostRegressor",
        "model_training_parameters": {
            "iterations": 2000,
            "depth": 7,
            "learning_rate": 0.02
        },
...
```

- [ ] **Step 2: Commit**

```bash
git add user_data/config.json
git commit -m "config: optimize FreqAI for Market Rent strategy"
```

---

### Task 4: Validation and Final Backtest

**Files:**
- Run: `freqtrade backtesting`

- [ ] **Step 1: Execute backtest over 30 days**

Run the backtest to verify the frequency and quality of trades across the 22 pairs.

Run: `freqtrade backtesting --config user_data/config.json --strategy StatArbStrategy --freqaimodel CatboostRegressor --timerange 20260325-20260425`

- [ ] **Step 2: Verify Results**
    *   Total trades > 15?
    *   Sortino > 3.0?
    *   Sharpe > 1.5?

- [ ] **Step 3: Final Commit**
