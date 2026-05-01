# Smart Money CVD Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the Market Rent Stat-Arb model by injecting Cumulative Volume Delta (CVD) as a proxy for institutional order flow ("Smart Money"), preventing false positive mean-reversion entries against strong directional trends.

**Architecture:** Use `taker_buy_base_volume` provided by Freqtrade to synthesize "Smart Money" flow, calculating the rolling CVD over 1h and 4h windows, and converting it to a rolling 24h z-score. Feed this to FreqAI.

**Tech Stack:** Freqtrade, FreqAI (CatBoost), Pandas.

---

### Task 1: Implement CVD Features in Strategy

**Files:**
- Modify: `user_data/strategies/StatArbStrategy.py`

- [ ] **Step 1: Add new hyperopt parameter for CVD Filter**

```python
    # Guardrails
    use_cvd_filter = DecimalParameter(0, 1, default=0.5, space='buy')
```

- [ ] **Step 2: Modify `feature_engineering_expand_basic` to add CVD features**

In `StatArbStrategy.py`, locate `feature_engineering_expand_basic` and add the CVD calculations.

```python
    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Stat-Arb specific features (Phase 2)
        """
        metadata = kwargs.get("metadata")
        dataframe = self.calculate_zscore(dataframe, metadata)
        
        # Funding Regime Detection (Market Rent)
        if 'funding_rate' in dataframe.columns:
            dataframe['%-funding_benefit_long'] = -dataframe['funding_rate']
            dataframe['%-funding_benefit_short'] = dataframe['funding_rate']
        else:
            dataframe['%-funding_benefit_long'] = 0
            dataframe['%-funding_benefit_short'] = 0

        # Microstructure integration: CVD Calculation
        if 'taker_buy_base_volume' in dataframe.columns:
             dataframe['%-ofi'] = (2 * dataframe['taker_buy_base_volume']) - dataframe['volume']
             taker_sell_base_volume = dataframe['volume'] - dataframe['taker_buy_base_volume']
             volume_delta = dataframe['taker_buy_base_volume'] - taker_sell_base_volume
             
             # CVD over 1h (12 * 5m) and 4h (48 * 5m)
             cvd_1h = volume_delta.rolling(window=12).sum()
             cvd_4h = volume_delta.rolling(window=48).sum()
             
             # 24h Z-score of CVD (288 * 5m)
             cvd_1h_mean = cvd_1h.rolling(window=288).mean()
             cvd_1h_std = cvd_1h.rolling(window=288).std()
             dataframe['%-cvd_zscore_1h'] = (cvd_1h - cvd_1h_mean) / cvd_1h_std
             
             cvd_4h_mean = cvd_4h.rolling(window=288).mean()
             cvd_4h_std = cvd_4h.rolling(window=288).std()
             dataframe['%-cvd_zscore_4h'] = (cvd_4h - cvd_4h_mean) / cvd_4h_std
             
             # Fill initial NaNs with 0
             dataframe['%-cvd_zscore_1h'] = dataframe['%-cvd_zscore_1h'].fillna(0)
             dataframe['%-cvd_zscore_4h'] = dataframe['%-cvd_zscore_4h'].fillna(0)
        else:
             dataframe['%-ofi'] = 0
             dataframe['%-cvd_zscore_1h'] = 0
             dataframe['%-cvd_zscore_4h'] = 0

        return dataframe
```

- [ ] **Step 3: Update `populate_entry_trend` with CVD Guardrails**

In `StatArbStrategy.py`, locate `populate_entry_trend`. Add the `use_cvd_filter` block right after the other filter conditions (like `use_ofi_filter`).

```python
        if self.use_cvd_filter.value > 0.5:
             enter_long_conditions.append(dataframe.get("%-cvd_zscore_1h", dataframe.get("%-cvd_zscore_1h_5m", 0)) > 0)
```

And for short:

```python
        if self.use_cvd_filter.value > 0.5:
             enter_short_conditions.append(dataframe.get("%-cvd_zscore_1h", dataframe.get("%-cvd_zscore_1h_5m", 0)) < 0)
```

- [ ] **Step 4: Update FreqAI Identifier in config**

Modify `user_data/config.json`. Change `"identifier": "market_rent_v12_balanced"` to `"market_rent_v2_smartmoney"`.

```json
        "identifier": "market_rent_v2_smartmoney",
```

- [ ] **Step 5: Run tests to verify the logic parses without errors**

Run: `pytest tests/test_strategy.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add user_data/strategies/StatArbStrategy.py user_data/config.json
git commit -m "feat: inject Smart Money CVD features and guardrail"
```

---

### Task 2: Validate Data and Run Backtest

**Files:**
- Run: `freqtrade backtesting`

- [ ] **Step 1: Test the FreqAI model training with new features**

Run a short backtest to ensure FreqAI processes the new features correctly without dropping them.
Run: `freqtrade backtesting --config user_data/config.json --strategy StatArbStrategy --freqaimodel CatboostRegressor --timerange 20260420-20260425`

- [ ] **Step 2: Commit any final tweaks if needed**

If the backtest requires minor adjustments (like NaN handling tweaks), apply and commit them.

```bash
git status
```
