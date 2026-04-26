#!/bin/bash
set -e
./venv/bin/freqtrade backtesting --config user_data/config.json --strategy StatArbStrategy --freqaimodel CatboostRegressor --timerange 20260401-20260426
