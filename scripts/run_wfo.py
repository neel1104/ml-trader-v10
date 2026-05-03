#!/usr/bin/env python3
import subprocess
import sys
import re
import argparse
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREQTRADE_BIN = PROJECT_ROOT / "venv" / "bin" / "freqtrade"
BASE_CONFIG_PATH = PROJECT_ROOT / "user_data" / "config.json"
TEMP_CONFIG_PATH = PROJECT_ROOT / "user_data" / "temp_wfo_config.json"
MODEL_DIR = PROJECT_ROOT / "user_data" / "models"

def run_command(cmd):
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
        if os.path.exists(TEMP_CONFIG_PATH):
            os.remove(TEMP_CONFIG_PATH)
        sys.exit(1)
    return output_str

def extract_metrics(output):
    sortino_match = re.search(r'Sortino\s+\|\s+([-\d.]+)', output)
    profit_match = re.search(r'Total profit %\s+\|\s+([-\d.]+)%', output)
    sortino = float(sortino_match.group(1)) if sortino_match else 0.0
    profit = float(profit_match.group(1)) if profit_match else 0.0
    return sortino, profit

def create_temp_config(identifier):
    with open(BASE_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    if 'freqai' not in config:
        config['freqai'] = {}
    config['freqai']['identifier'] = identifier
    # Hard purge to prevent cache issues
    config['freqai']['purge_old_models'] = True
    with open(TEMP_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    return str(TEMP_CONFIG_PATH)

def main():
    run_ts = datetime.now().strftime("%H%M%S")
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization (WFO)")
    parser.add_argument("--start", type=str, default="20260301", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="20260425", help="End date YYYYMMDD")
    parser.add_argument("--train-days", type=int, default=14, help="In-sample training days")
    parser.add_argument("--test-days", type=int, default=7, help="Out-of-sample testing days")
    parser.add_argument("--epochs", type=int, default=30, help="Hyperopt epochs per window")
    parser.add_argument("--base-id", type=str, default=f"wfo_{run_ts}", help="FreqAI base identifier")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y%m%d")
    end_date = datetime.strptime(args.end, "%Y%m%d")
    current_date = start_date
    results = []
    
    print(f"Starting Bulletproof WFO from {args.start} to {args.end} | Run ID: {args.base_id}")
    print("-" * 50)
    
    while current_date + timedelta(days=args.train_days + args.test_days) <= end_date:
        train_start = current_date.strftime("%Y%m%d")
        train_end = (current_date + timedelta(days=args.train_days)).strftime("%Y%m%d")
        test_start = train_end
        test_end = (current_date + timedelta(days=args.train_days + args.test_days)).strftime("%Y%m%d")
        
        window_id = f"{args.base_id}_{train_start}"
        print(f"\n>>> WINDOW: Train [{train_start}-{train_end}] | Test [{test_start}-{test_end}] | ID: {window_id}")
        
        # PERMANENT FIX: Delete existing model folder to prevent any cache reuse
        target_dir = MODEL_DIR / window_id
        if target_dir.exists():
            print(f"Wiping existing model directory: {target_dir}")
            shutil.rmtree(target_dir)

        config_path = create_temp_config(window_id)
        
        # 1. Hyperopt
        hyperopt_cmd = [str(FREQTRADE_BIN), "hyperopt", "--config", config_path, "--strategy", "StatArbStrategy", 
                        "--freqaimodel", "CatboostRegressor", "--hyperopt-loss", "SortinoHyperOptLoss", 
                        "--spaces", "buy", "sell", "--timerange", f"{train_start}-{train_end}", "-e", str(args.epochs), "-j", "1"]
        run_command(hyperopt_cmd)
        
        # 2. IS Backtest
        is_backtest_cmd = [str(FREQTRADE_BIN), "backtesting", "--config", config_path, "--strategy", "StatArbStrategy", 
                           "--freqaimodel", "CatboostRegressor", "--timerange", f"{train_start}-{train_end}"]
        is_output = run_command(is_backtest_cmd)
        is_sortino, is_profit = extract_metrics(is_output)
        
        # 3. OOS Backtest
        oos_backtest_cmd = [str(FREQTRADE_BIN), "backtesting", "--config", config_path, "--strategy", "StatArbStrategy", 
                            "--freqaimodel", "CatboostRegressor", "--timerange", f"{test_start}-{test_end}"]
        oos_output = run_command(oos_backtest_cmd)
        oos_sortino, oos_profit = extract_metrics(oos_output)
        
        results.append({"window": f"{test_start}-{test_end}", "is_profit": is_profit, "oos_profit": oos_profit, "oos_sortino": oos_sortino})
        print(f"Result: IS Profit: {is_profit:.2f}% | OOS Profit: {oos_profit:.2f}% | OOS Sortino: {oos_sortino:.2f}")
        current_date += timedelta(days=args.test_days)

    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)

    print("\n" + "=" * 50 + "\nWFO RESULTS SUMMARY\n" + "=" * 50)
    print(f"{'Test Window':<20} | {'IS Profit':<10} | {'OOS Profit':<10} | {'OOS Sortino':<10}\n" + "-" * 55)
    total_is = total_oos = 0
    for r in results:
        print(f"{r['window']:<20} | {r['is_profit']:>9.2f}% | {r['oos_profit']:>9.2f}% | {r['oos_sortino']:>9.2f}")
        total_is += r['is_profit']
        total_oos += r['oos_profit']
    wfe = (total_oos / total_is * 100) if total_is > 0 else 0
    print("-" * 55 + f"\nTotal IS Profit:  {total_is:.2f}%\nTotal OOS Profit: {total_oos:.2f}%\nWalk-Forward Efficiency (WFE): {wfe:.1f}%")
    print(">>> EDGE VALIDATED" if wfe > 50 else ">>> EDGE FAILED")

if __name__ == "__main__":
    main()
