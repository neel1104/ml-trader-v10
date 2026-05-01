# Design Doc: Market Rent Stat-Arb Evolution

**Date:** 2026-05-01
**Status:** Approved
**Goal:** Transform the `StatArbStrategy` from a pure price-divergence model into an institutional-grade "Market Rent" yield harvester, targeting Sortino > 3.0 and Sharpe > 1.5.

## 1. Problem Statement
The current version of `StatArbStrategy` is too conservative (2 trades/month) and ignores structural "market rent" (funding rates). It relies solely on price anomalies across only 4 pairs, leading to low capital efficiency.

## 2. Architecture: Hub-and-Spoke Anchor System
- **Universal Anchor:** `BTC/USDT:USDT` acts as the primary benchmark for all relative value calculations.
- **Log-Spread Calculation:**
  - `spread = log(Price_Asset) - log(Price_BTC)`
  - Rolling Z-score calculated over a `lookback_period` (hyperoptable, default 168h).
- **FreqAI Integration:** CatBoost model predicts the Z-score value 3 candles ahead (`&-zscore_target`) to filter for high-conviction mean-reversion.

## 3. Core Logic: Regime-Based Funding Harvest
The strategy will dynamically adjust the `zscore_entry` threshold based on the net funding benefit of the trade.

### 3.1. Funding Regimes
| Regime | Condition (Net Funding for Position) | Z-Score Threshold | Intent |
| :--- | :--- | :--- | :--- |
| **Favorable** | `Benefit > 0.01%` per 8h | **2.0** (Aggressive) | Harvest yield + price reversion. |
| **Neutral** | `-0.01% to 0.01%` | **2.5** (Standard) | Pure statistical arbitrage. |
| **Penalty** | `Cost > 0.01%` per 8h | **3.5** (Extreme) | Protect principal from "carry cost." |

### 3.2. Signal Stacking (Phase 3 RESEARCH.md)
Entries are only permitted if:
1. **Regime Threshold** is met.
2. **FreqAI Direction** aligns: `&-zscore_target` confirms reversion towards mean.
3. **Magnitude Check:** Predicted reversion magnitude > `min_predicted_magnitude`.

## 4. Extended Blue-Chip Universe
Expanding the whitelist to 22 high-liquidity assets to increase trade frequency without compromising quality:
`BTC, ETH, SOL, AVAX, MATIC, LINK, ADA, DOT, ATOM, NEAR, LTC, BCH, XRP, TRX, UNI, APT, ARB, OP, SUI, INJ, RNDR, FET`.

## 5. Implementation Details
- **Files:**
  - `user_data/strategies/StatArbStrategy.py`: Update entry logic and regime detection.
  - `user_data/config.json`: Expand pairlist and update FreqAI parameters.
- **Data:** Ensure `download-data` includes funding rate history for all 22 pairs.

## 6. Success Criteria
- **Sortino Ratio:** > 3.0
- **Sharpe Ratio:** > 1.5
- **Trade Frequency:** > 10 trades per month (across the universe).
- **Max Drawdown:** < 10%

## 7. Self-Review Notes
- **Placeholder Scan:** All thresholds are defined or hyperoptable.
- **Consistency:** Anchoring all pairs to BTC ensures a unified market-neutral beta.
- **Scope:** This is a focused evolution of the existing strategy, well-within implementation capacity.
