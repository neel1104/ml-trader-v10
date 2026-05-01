# Design Doc: Smart Money Data Injection (Phase 3)

**Date:** 2026-05-01
**Status:** Approved
**Goal:** Enhance the Market Rent Stat-Arb model by injecting Cumulative Volume Delta (CVD) as a proxy for institutional order flow ("Smart Money"), preventing false positive mean-reversion entries against strong directional trends.

## 1. Problem Statement
Despite achieving an 83% win rate in the static backtest, the Market Rent strategy relies entirely on price-derived z-scores and funding rates. It lacks structural awareness of the underlying order book liquidity. By injecting a proxy for "Smart Money" accumulation or distribution, the FreqAI model can better differentiate between a temporary liquidity void (good for mean reversion) and sustained institutional selling (bad for mean reversion).

## 2. Architecture: Microstructure Proxy
- **Data Source:** We will synthesize "Smart Money" flow using the existing Binance futures data, specifically the `taker_buy_base_volume` provided by Freqtrade.
- **Metric:** Cumulative Volume Delta (CVD).
  - `taker_sell_base_volume = volume - taker_buy_base_volume`
  - `volume_delta = taker_buy_base_volume - taker_sell_base_volume`
  - `CVD = rolling_sum(volume_delta)` over specified windows (1h, 4h).
- **Normalization:** To make the feature asset-agnostic, the CVD will be converted into a rolling Z-score over a 24h period (`%-cvd_zscore_1h`, `%-cvd_zscore_4h`).

## 3. Model Integration (FreqAI)
- **Features:** The normalized CVD features will be appended to the `feature_engineering_expand_basic` pipeline. The CatBoost model will naturally weigh these features alongside existing RSI/MFI/Volatility metrics.
- **Model Isolation:** To preserve the existing `market_rent` models, we will update the `identifier` in `config.json` to `market_rent_v2_smartmoney`.

## 4. Strategy Guardrails (Signal Stacking)
- Introduce a new hyperoptable boolean parameter: `use_cvd_filter`.
- If enabled (value > 0.5), entries will be hard-gated by the 1h CVD direction:
  - **Long:** `%-cvd_zscore_1h > 0` (Smart money is buying).
  - **Short:** `%-cvd_zscore_1h < 0` (Smart money is selling).
- This allows Bayesian Optimization to determine whether a hard rule or soft feature-weighting yields a superior Sharpe ratio.

## 5. Success Criteria
- **Implementation:** CVD features are successfully calculated without NaNs in the training set.
- **Backtesting:** Model trains successfully with the new features on the 22-asset universe.

## 6. Self-Review Notes
- **Scope:** The feature is scoped cleanly to `StatArbStrategy.py` and `config.json`.
- **Dependencies:** Relies entirely on data already fetched via `download-data`. No external APIs required.
