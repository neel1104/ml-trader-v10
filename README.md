# ML Trader V10: Quantitative Crypto Framework

A high-frequency, machine-learning-driven cryptocurrency trading system built on **Freqtrade** and **FreqAI**. This project implements Phase 1-3 of a quantitative research roadmap, focusing on **CatBoost** models, market microstructure, and "Smart Money" tick-level features.

## 🚀 Quick Start

### 1. Setup Environment
Ensure you have Python 3.10+ installed.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data (Including Tick/Trades)
Download 60 days of historical data, including raw transaction data for Smart Money features.
```bash
./scripts/download_data.sh
```

### 3. Run Backtest
Train the CatBoost model and run a backtest simulation using current parameters.
```bash
./scripts/run_backtest.sh
```

### 4. Optimize Strategy
Find the optimal risk-adjusted entry/exit thresholds using **Bayesian Optimization (Optuna)** targeting the **Sortino Ratio**.
```bash
./scripts/run_hyperopt.sh
```

## 🛠 Project Structure

- `user_data/strategies/CatBoostStrategy.py`: The core FreqAI strategy with microstructure and OFI.
- `user_data/freqaimodels/CatboostRegressor.py`: Custom CatBoost model implementation.
- `user_data/hyperopts/SortinoHyperOptLoss.py`: Custom loss function for risk-adjusted optimization.
- `user_data/config.json`: Main configuration (Futures mode, CCXT settings, FreqAI parameters).
- `scripts/`: Helper scripts for data download, backtesting, and hyperopt.
- `research.md`: The foundational quantitative research paper for this project.

## 🧠 Strategy Highlights (Phases 1-3)

- **Model:** CatBoost Regressor with optimal Selectivity.
- **Smart Money Features:** 
    - **True OFI:** Trade Flow Imbalance calculated from raw tick data.
    - **Whale Tracking:** Isolating large institutional trade volumes (top 5th percentile).
- **Microstructure Features:** 
    - Bid/Ask Spread, Price-Range, and Mock-CVD.
- **Target:** 3-candle forward returns.
- **Optimization:** Sortino Ratio-driven (penalizing downside volatility over absolute profit).
- **Trading Mode:** Binance Futures (Long/Short enabled).

## 🧪 Testing
Run the comprehensive 6-test suite to verify the framework:
```bash
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/pytest tests/ -v
```
