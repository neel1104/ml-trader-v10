# Initialize Project and Setup Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Setup the basic project structure, dependencies, and Freqtrade configuration.

**Architecture:** Initialize `requirements.txt` for dependency management, `user_data/config.json` for Freqtrade configuration, and `tests/test_config.py` for verification.

**Tech Stack:** Freqtrade, Pytest, Python.

---

### Task 1: Create dependencies file

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**
```text
freqtrade>=2024.2
freqtrade[freqai]>=2024.2
pytest>=8.0.0
catboost>=1.2.0
```

- [ ] **Step 2: Commit**
```bash
git add requirements.txt
git commit -m "chore: add project dependencies"
```

### Task 2: Create Freqtrade configuration

**Files:**
- Create: `user_data/config.json`

- [ ] **Step 1: Create user_data directory**
```bash
mkdir -p user_data
```

- [ ] **Step 2: Create user_data/config.json**
```json
{
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "dry_run": true,
    "dry_run_wallet": 1000,
    "cancel_open_orders_on_exit": false,
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "ccxt_config": {},
        "ccxt_async_config": {},
        "pair_whitelist": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "pair_blacklist": []
    },
    "pairlists": [
        {"method": "StaticPairList"}
    ],
    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 15,
        "backtest_period_days": 7,
        "identifier": "catboost_model_v1",
        "feature_parameters": {
            "include_timeframes": ["5m", "15m", "1h"],
            "include_corr_pairlists": ["BTC/USDT", "ETH/USDT"]
        },
        "data_split_parameters": {
            "test_size": 0.25
        }
    }
}
```

- [ ] **Step 3: Commit**
```bash
git add user_data/config.json
git commit -m "chore: add freqtrade configuration"
```

### Task 3: Write test for configuration loading

**Files:**
- Create: `tests/test_config.py`

- [ ] **Step 1: Create tests directory**
```bash
mkdir -p tests
```

- [ ] **Step 2: Write tests/test_config.py**
```python
import json
import os

def test_config_validity():
    config_path = "user_data/config.json"
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    assert config["dry_run"] is True
    assert config["freqai"]["enabled"] is True
    assert "BTC/USDT" in config["exchange"]["pair_whitelist"]
```

- [ ] **Step 3: Run test to verify it passes**
Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add tests/test_config.py
git commit -m "test: add configuration validity test"
```
