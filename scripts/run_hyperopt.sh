#!/bin/bash
set -e
./venv/bin/freqtrade hyperopt --config user_data/config.json --strategy StatArbStrategy --freqaimodel CatboostRegressor --timerange 20260401-20260426 --hyperopt-loss SortinoHyperOptLoss --spaces buy sell -e 50
