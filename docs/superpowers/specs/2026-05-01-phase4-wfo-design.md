# Design Doc: Phase 4 Walk-Forward Optimization (WFO) Pipeline

**Date:** 2026-05-01
**Status:** Approved
**Goal:** Build a Walk-Forward Optimization (WFO) pipeline to rigorously test the Market Rent Stat-Arb model's edge across rolling time windows. This prevents static curve-fitting and calculates the Walk-Forward Efficiency (WFE) score as recommended in Phase 4 of `RESEARCH.md`.

## 1. Problem Statement
The strategy currently holds an 83% win rate in a static backtest. However, cryptocurrency markets are highly non-stationary. A static hyperparameter optimization over a single 30-day block often produces an overfit model that collapses in forward testing. To ensure institutional-grade reliability, we must continuously retune parameters on an in-sample window and test them on an unseen out-of-sample window, calculating the strategy's true robustness (WFE).

## 2. Architecture: Python Orchestrator
- **File:** `scripts/run_wfo.py`
- **Language:** Python 3.10+
- **Reasoning:** Python is required to parse the JSON outputs from Freqtrade (`.last_result.json` and backtest meta files) to extract the Sortino Ratio and Total Profit accurately. A bash script would be too fragile for complex JSON parsing and math operations.

## 3. The WFO Loop Logic
The script will orchestrate Freqtrade subprocesses over rolling dates.
- **Data Range:** Defaults to the last 60 days of downloaded data.
- **In-Sample (Train/Optimize):** 14-day window.
  - Action: Execute `freqtrade hyperopt --timerange <in-sample-dates> --hyperopt-loss SortinoHyperOptLoss`
  - Extract: The best parameters are automatically saved to `StatArbStrategy.json`.
- **Out-of-Sample (Test):** 7-day window immediately following the In-Sample period.
  - Action: Execute `freqtrade backtesting --timerange <out-of-sample-dates>`
  - Extract: Total Profit % and Sortino Ratio from the JSON backtest results.
- **Roll Forward:** Slide the In-Sample start date forward by 7 days. Repeat until the end of the available data is reached.

## 4. WFE Calculation and Output
- **Walk-Forward Efficiency (WFE):** 
  `WFE = (Average Out-of-Sample Return / Average In-Sample Return) * 100`
- A score > 50% indicates the edge is robust.
- The script will output a summarized Markdown table showing the In-Sample and Out-of-Sample performance for every rolling window, concluding with the final WFE score.

## 5. Implementation Scope
- Create `scripts/run_wfo.py`.
- Update `README.md` to document the new optimization pipeline command.

## 6. Self-Review Notes
- **Dependencies:** The script will strictly use standard library (`json`, `subprocess`, `datetime`, `argparse`) or existing requirements (like `pandas` if necessary). No new external pip packages needed.
- **Isolation:** This script orchestrates Freqtrade but does not modify the `StatArbStrategy.py` logic. It tests what currently exists.