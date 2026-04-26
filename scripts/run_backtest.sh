#!/bin/bash
set -e
./venv/bin/freqtrade backtesting --config user_data/config.json --strategy CatBoostStrategy --freqaimodel CatboostRegressor --timerange 20260301-20260426
