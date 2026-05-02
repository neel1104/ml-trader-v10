# Phase 4 Walk-Forward Optimization (WFO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a Walk-Forward Optimization (WFO) pipeline to rigorously test the Market Rent Stat-Arb model's edge across rolling time windows.

**Architecture:** A Python script (`scripts/run_wfo.py`) that orchestrates `freqtrade hyperopt` and `freqtrade backtesting` subprocesses over rolling 14-day train and 7-day test windows, parsing STDOUT with Regex to extract Sortino and Profit metrics, and calculating the Walk-Forward Efficiency (WFE).

**Tech Stack:** Python 3.10+, Subprocess, Regex (re), Datetime.

---

### Task 1: Create WFO Orchestrator Script

**Files:**
- Create: `scripts/run_wfo.py`
- Modify: `README.md` (to document the new command)

- [x] **Step 1: Write the WFO Python Script**

Create `scripts/run_wfo.py` with the following complete code:

```python
#!/usr/bin/env python3
import subprocess
import re
import argparse
from datetime import datetime, timedelta

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command:\n{result.stderr}")
    return result.stdout

def extract_metrics(output):
    sortino_match = re.search(r'Sortino\s+\|\s+([-\d.]+)', output)
    profit_match = re.search(r'Total profit %\s+\|\s+([-\d.]+)%', output)
    
    sortino = float(sortino_match.group(1)) if sortino_match else 0.0
    profit = float(profit_match.group(1)) if profit_match else 0.0
    return sortino, profit

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization (WFO)")
    parser.add_argument("--start", type=str, default="20260301", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="20260425", help="End date YYYYMMDD")
    parser.add_argument("--train-days", type=int, default=14, help="In-sample training days")
    parser.add_argument("--test-days", type=int, default=7, help="Out-of-sample testing days")
    parser.add_argument("--epochs", type=int, default=30, help="Hyperopt epochs per window")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y%m%d")
    end_date = datetime.strptime(args.end, "%Y%m%d")
    
    current_date = start_date
    results = []
    
    print(f"Starting WFO from {args.start} to {args.end}")
    print("-" * 50)
    
    while current_date + timedelta(days=args.train_days + args.test_days) <= end_date:
        train_start = current_date.strftime("%Y%m%d")
        train_end = (current_date + timedelta(days=args.train_days)).strftime("%Y%m%d")
        
        test_start = train_end
        test_end = (current_date + timedelta(days=args.train_days + args.test_days)).strftime("%Y%m%d")
        
        print(f"\\n>>> Window: Train [{train_start}-{train_end}] | Test [{test_start}-{test_end}]")
        
        # 1. Run Hyperopt (In-Sample)
        hyperopt_cmd = (
            f"./venv/bin/freqtrade hyperopt --config user_data/config.json "
            f"--strategy StatArbStrategy --freqaimodel CatboostRegressor "
            f"--hyperopt-loss SortinoHyperOptLoss --spaces buy sell "
            f"--timerange {train_start}-{train_end} -e {args.epochs} -j 1"
        )
        run_command(hyperopt_cmd)
        
        # We don't easily extract in-sample profit from hyperopt stdout without complex parsing of the best epoch.
        # Instead, we run a quick backtest over the train period to get the exact In-Sample baseline.
        is_backtest_cmd = (
            f"./venv/bin/freqtrade backtesting --config user_data/config.json "
            f"--strategy StatArbStrategy --freqaimodel CatboostRegressor "
            f"--timerange {train_start}-{train_end}"
        )
        is_output = run_command(is_backtest_cmd)
        is_sortino, is_profit = extract_metrics(is_output)
        
        # 2. Run Backtest (Out-of-Sample)
        oos_backtest_cmd = (
            f"./venv/bin/freqtrade backtesting --config user_data/config.json "
            f"--strategy StatArbStrategy --freqaimodel CatboostRegressor "
            f"--timerange {test_start}-{test_end}"
        )
        oos_output = run_command(oos_backtest_cmd)
        oos_sortino, oos_profit = extract_metrics(oos_output)
        
        results.append({
            "window": f"{test_start}-{test_end}",
            "is_profit": is_profit,
            "oos_profit": oos_profit,
            "oos_sortino": oos_sortino
        })
        
        print(f"Result: IS Profit: {is_profit:.2f}% | OOS Profit: {oos_profit:.2f}% | OOS Sortino: {oos_sortino:.2f}")
        
        # Roll forward
        current_date += timedelta(days=args.test_days)

    print("\\n" + "=" * 50)
    print("WFO RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Test Window':<20} | {'IS Profit':<10} | {'OOS Profit':<10} | {'OOS Sortino':<10}")
    print("-" * 55)
    
    total_is = 0
    total_oos = 0
    
    for r in results:
        print(f"{r['window']:<20} | {r['is_profit']:>9.2f}% | {r['oos_profit']:>9.2f}% | {r['oos_sortino']:>9.2f}")
        total_is += r['is_profit']
        total_oos += r['oos_profit']
        
    wfe = (total_oos / total_is * 100) if total_is > 0 else 0
    
    print("-" * 55)
    print(f"Total IS Profit:  {total_is:.2f}%")
    print(f"Total OOS Profit: {total_oos:.2f}%")
    print(f"Walk-Forward Efficiency (WFE): {wfe:.1f}%")
    
    if wfe > 50:
        print(">>> EDGE VALIDATED: WFE is > 50%. Strategy is robust.")
    else:
        print(">>> EDGE FAILED: WFE is < 50%. Strategy is likely overfit.")

if __name__ == "__main__":
    main()
```

- [x] **Step 2: Make the script executable**

Run: `chmod +x scripts/run_wfo.py`

- [x] **Step 3: Update README.md**

Modify `README.md`. Under the `## 🧪 Testing` section, add a new section for Phase 4 WFO:

```markdown
## 🔍 Phase 4: Walk-Forward Optimization (WFO)
To validate the strategy's robustness and calculate the Walk-Forward Efficiency (WFE) score, run the WFO pipeline. This will train on 14-day rolling windows and test on unseen 7-day forward windows.

```bash
./scripts/run_wfo.py --start 20260301 --end 20260425 --epochs 30
```
```

- [x] **Step 4: Commit**

```bash
git add scripts/run_wfo.py README.md
git commit -m "feat: add Walk-Forward Optimization (WFO) orchestrator script"
```
