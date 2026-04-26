# ML Trader V10: Quantitative Crypto Framework

A high-frequency, machine-learning-driven cryptocurrency trading system built on **Freqtrade** and **FreqAI**. This project implements Phase 1 of a quantitative research roadmap, focusing on **CatBoost** models and market microstructure features.

## 🚀 Quick Start

### 1. Setup Environment
Ensure you have Python 3.10+ installed.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data
Download historical data for the configured pairs (BTC, ETH, SOL) across multiple timeframes.
```bash
./scripts/download_data.sh
```

### 3. Run Backtest
Train the CatBoost model and run a backtest simulation.
```bash
./scripts/run_backtest.sh
```

## 🛠 Project Structure

- `user_data/strategies/CatBoostStrategy.py`: The core FreqAI strategy.
- `user_data/config.json`: Project configuration (Futures mode, CatBoost parameters).
- `scripts/`: Helper scripts for data management and backtesting.
- `tests/`: Pytest suite for configuration and strategy validation.
- `research.md`: The foundational quantitative research paper for this project.

## 🧠 Strategy Highlights

- **Model:** CatBoost Regressor (via FreqAI).
- **Features:** 
    - **Microstructure:** Spread, Price-Range, and Mock-CVD.
    - **Technical:** RSI, MFI, ADX (Multiple periods).
    - **Temporal:** Day of week, Hour of day.
- **Target:** 3-candle forward returns.
- **Trading Mode:** Futures (Long/Short enabled).

## 🧪 Testing
Run the test suite to ensure everything is configured correctly:
```bash
./venv/bin/pytest tests/ -v
```
