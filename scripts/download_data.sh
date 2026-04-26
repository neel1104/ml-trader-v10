#!/bin/bash
set -e
freqtrade download-data --config user_data/config.json --days 60 -t 5m 15m 1h
