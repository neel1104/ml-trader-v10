# StatArbStrategy Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `StatArbStrategy` from a slightly negative post-fee strategy into an institutional-grade framework (Sortino > 3.0, Sharpe > 1.5) by reducing fee impact and implementing high-conviction signal stacking.

**Architecture:** We will transition to "Signal Stacking" as per Phase 1 & 3 of `RESEARCH.md`. This involves adding technical and microstructure filters to the entry logic to ensure only high-probability, high-magnitude divergence events are traded, thereby increasing the average profit per trade to comfortably cover fees.

**Tech Stack:** Freqtrade, FreqAI (CatBoost), TALib, Pandas.

---

### Task 1: Implement Signal Stacking in Strategy Logic

**Files:**
- Modify: `user_data/strategies/StatArbStrategy.py`

- [ ] **Step 1: Add microstructure and technical filters to entry trend**

Modify `populate_entry_trend` to include `%-ofi`, `%-rsi`, and `%-mfi` filters.

```python
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Define high-conviction stacking conditions
        # Long Entry: Z-score low + predicted rise + RSI/MFI oversold + Positive OFI
        enter_long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["%-zscore"] < -self.zscore_entry.value,
            dataframe["&-zscore_target"] > dataframe["%-zscore"],
            dataframe["%-rsi-14"] < 40, # RSI filter
            dataframe["%-mfi-14"] < 40, # MFI filter
            dataframe["%-ofi"] > 0       # Positive Order Flow Imbalance
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "stat_arb_long_stacked")

        # Short Entry: Z-score high + predicted fall + RSI/MFI overbought + Negative OFI
        enter_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["%-zscore"] > self.zscore_entry.value,
            dataframe["&-zscore_target"] < dataframe["%-zscore"],
            dataframe["%-rsi-14"] > 60,
            dataframe["%-mfi-14"] > 60,
            dataframe["%-ofi"] < 0
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "stat_arb_short_stacked")

        return dataframe
```

- [ ] **Step 2: Commit**

```bash
git add user_data/strategies/StatArbStrategy.py
git commit -m "strat: implement signal stacking with RSI/MFI/OFI filters"
```

---

### Task 2: Increase Target Magnitude and Model Complexity

**Files:**
- Modify: `user_data/config.json`

- [ ] **Step 1: Update FreqAI parameters for high-magnitude signals**

Increase iteration count and reduce learning rate for better convergence. Update the identifier to force re-training.

```json
    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 30,
        "backtest_period_days": 7,
        "identifier": "stat_arb_stacked_v1",
        "model_name": "CatboostRegressor",
        "model_training_parameters": {
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.01
        },
...
```

- [ ] **Step 2: Commit**

```bash
git add user_data/config.json
git commit -m "config: increase model complexity and iterations for stacked strategy"
```

---

### Task 3: Multi-Week Hyperopt for Institutional Goals

**Files:**
- Modify: `user_data/strategies/StatArbStrategy.py` (ensure DecimalParameter ranges are wide enough)

- [ ] **Step 1: Run hyperopt over a 30-day timerange**

Run hyperopt targeting the `SortinoHyperOptLoss` to find thresholds that yield Sortino > 3.0.

Run: `freqtrade hyperopt --config user_data/config.json --strategy StatArbStrategy --hyperopt-loss SortinoHyperOptLoss --freqaimodel CatboostRegressor --timerange 20260320-20260420 --spaces buy sell --epochs 100 -j 1`

- [ ] **Step 2: Update Strategy with optimized parameters**

Verify `StatArbStrategy.json` is updated and rerun backtest.

- [ ] **Step 3: Commit**

```bash
git add user_data/strategies/StatArbStrategy.json
git commit -m "opt: update strategy with 30-day optimized stacked parameters"
```

---

### Task 4: Final Validation Against Research Goals

**Files:**
- Run: `freqtrade backtesting`

- [ ] **Step 1: Execute final backtest and evaluate ratios**

Run: `freqtrade backtesting --config user_data/config.json --strategy StatArbStrategy --freqaimodel CatboostRegressor --timerange 20260401-20260426`

- [ ] **Step 2: Verify goals**
    *   Sortino > 3.0?
    *   Sharpe > 1.5?
    *   Drawdown < 15%?

- [ ] **Step 3: Commit final results summary**
