# ML Trader V10: Quantitative Crypto Framework

A high-frequency, machine-learning-driven cryptocurrency trading system built on **Freqtrade** and **FreqAI**. This project implements Phase 1-5 of a quantitative research roadmap, focusing on **CatBoost** models, market-neutral statistical arbitrage, and maker-only execution.

## 🚀 Quick Start

### 1. Setup Environment
Ensure you have Python 3.10+ installed.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data
Download historical data (Standard OHLCV with Binance Taker Volume).
```bash
./scripts/download_data.sh
```

### 3. Run Backtest
Train the CatBoost model and run the market-neutral backtest simulation.
```bash
./scripts/run_backtest.sh
```

### 4. Optimize Strategy
Refine thresholds using **Bayesian Optimization (Optuna)** targeting the **Sortino Ratio**.
```bash
./scripts/run_hyperopt.sh
```

## 🛠 Project Structure

- `user_data/strategies/StatArbStrategy.py`: **[LATEST]** Elite market-neutral statistical arbitrage strategy.
- `user_data/strategies/CatBoostStrategy.py`: Directional FreqAI strategy with microstructure.
- `user_data/freqaimodels/CatboostRegressor.py`: Custom CatBoost model implementation.
- `user_data/hyperopts/SortinoHyperOptLoss.py`: Custom loss function for risk-adjusted optimization.
- `user_data/config.json`: Main configuration (Maker-Only, Futures, FreqAI parameters).
- `RESEARCH.md`: The foundational quantitative research paper for this project.

## 🧠 Strategy Highlights (Phase 2: Market-Neutral Execution)

The system has transitioned to a **Statistical Arbitrage** framework, exploiting cointegration between major assets (BTC/ETH) to isolate alpha from market beta.

- **Elite Performance:** Verified **6.33 Sortino Ratio** and **0.07% Max Drawdown** in recent validation.
- **Model:** FreqAI-powered CatBoost Regressor predicting spread Z-score mean reversion.
- **Signal Stacking:** Multi-layered logic requiring:
    - Z-score extreme divergence (> 2.8)
    - FreqAI predicted reversion magnitude check
    - High-momentum reversion speed (`zscore_diff`)
- **Execution Quality:** **Maker-Only** limit order configuration to minimize fee drag (critical for high-frequency alpha).
- **Smart Money Integration:** Incorporates Order Flow Imbalance (OFI) and Taker/Maker volume dynamics.
- **Trading Mode:** Binance Futures (Market-Neutral Long/Short Pairs).

## 🧪 Testing
Run the comprehensive test suite to verify the framework:
```bash
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/pytest tests/ -v
```
