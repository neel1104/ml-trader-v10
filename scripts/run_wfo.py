#!/usr/bin/env python3
import subprocess
import sys
import re
import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREQTRADE_BIN = PROJECT_ROOT / "venv" / "bin" / "freqtrade"
BASE_CONFIG_PATH = PROJECT_ROOT / "user_data" / "config.json"
TEMP_CONFIG_PATH = PROJECT_ROOT / "user_data" / "temp_wfo_config.json"

def run_command(cmd):
    # cmd is expected to be a list
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    full_output = []
    for line in process.stdout:
        print(line, end="")
        full_output.append(line)
    
    process.wait()
    output_str = "".join(full_output)
    
    if process.returncode != 0:
        print(f"Error running command. Exit code: {process.returncode}")
        # Clean up temp config before exiting
        if os.path.exists(TEMP_CONFIG_PATH):
            os.remove(TEMP_CONFIG_PATH)
        sys.exit(1)
    return output_str

def extract_metrics(output):
    # Match profit like "Total profit % | 1.23%"
    # Match sortino like "Sortino | 2.5"
    sortino_match = re.search(r'Sortino\s+\|\s+([-\d.]+)', output)
    profit_match = re.search(r'Total profit %\s+\|\s+([-\d.]+)%', output)
    
    sortino = float(sortino_match.group(1)) if sortino_match else 0.0
    profit = float(profit_match.group(1)) if profit_match else 0.0
    return sortino, profit

def create_temp_config(identifier):
    with open(BASE_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Update the identifier for this specific window
    if 'freqai' not in config:
        config['freqai'] = {}
    config['freqai']['identifier'] = identifier
    
    with open(TEMP_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    
    return str(TEMP_CONFIG_PATH)

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization (WFO) - Robust Version")
    parser.add_argument("--start", type=str, default="20260301", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="20260425", help="End date YYYYMMDD")
    parser.add_argument("--train-days", type=int, default=14, help="In-sample training days")
    parser.add_argument("--test-days", type=int, default=7, help="Out-of-sample testing days")
    parser.add_argument("--epochs", type=int, default=30, help="Hyperopt epochs per window")
    parser.add_argument("--base-id", type=str, default="market_rent_wfo", help="FreqAI base identifier")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y%m%d")
    end_date = datetime.strptime(args.end, "%Y%m%d")
    
    current_date = start_date
    results = []
    
    print(f"Starting Robust WFO from {args.start} to {args.end}")
    print("-" * 50)
    
    while current_date + timedelta(days=args.train_days + args.test_days) <= end_date:
        train_start = current_date.strftime("%Y%m%d")
        train_end = (current_date + timedelta(days=args.train_days)).strftime("%Y%m%d")
        
        test_start = train_end
        test_end = (current_date + timedelta(days=args.train_days + args.test_days)).strftime("%Y%m%d")
        
        window_id = f"{args.base_id}_{train_start}"
        
        print(f"\n>>> WINDOW: Train [{train_start}-{train_end}] | Test [{test_start}-{test_end}] | ID: {window_id}")
        
        # 1. Create temp config with the unique identifier
        config_path = create_temp_config(window_id)
        
        # 2. Run Hyperopt (In-Sample)
        hyperopt_cmd = [
            str(FREQTRADE_BIN), "hyperopt",
            "--config", config_path,
            "--strategy", "StatArbStrategy",
            "--freqaimodel", "CatboostRegressor",
            "--hyperopt-loss", "SortinoHyperOptLoss",
            "--spaces", "buy", "sell",
            "--timerange", f"{train_start}-{train_end}",
            "-e", str(args.epochs),
            "-j", "1"
        ]
        run_command(hyperopt_cmd)
        
        # 3. Run Backtest (In-Sample Baseline)
        is_backtest_cmd = [
            str(FREQTRADE_BIN), "backtesting",
            "--config", config_path,
            "--strategy", "StatArbStrategy",
            "--freqaimodel", "CatboostRegressor",
            "--timerange", f"{train_start}-{train_end}"
        ]
        is_output = run_command(is_backtest_cmd)
        is_sortino, is_profit = extract_metrics(is_output)
        
        # 4. Run Backtest (Out-of-Sample)
        oos_backtest_cmd = [
            str(FREQTRADE_BIN), "backtesting",
            "--config", config_path,
            "--strategy", "StatArbStrategy",
            "--freqaimodel", "CatboostRegressor",
            "--timerange", f"{test_start}-{test_end}"
        ]
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

    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)

    print("\n" + "=" * 50)
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
