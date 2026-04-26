#!/bin/bash
set -e
./venv/bin/freqtrade hyperopt --config user_data/config.json --strategy CatBoostStrategy --freqaimodel CatboostRegressor --timerange 20260301-20260426 --hyperopt-loss SortinoHyperOptLoss --spaces buy sell -e 50
