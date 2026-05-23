import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from freqtrade.configuration import Configuration
from user_data.freqaimodels.CatboostRegressor import CatboostRegressor

def get_mock_model():
    config = Configuration.from_files(["user_data/config.json"])
    # Set necessary overrides for test isolation
    config["timeframe"] = "5m"
    config["freqai"]["purge_old_models"] = True
    config["freqai"]["data_split_parameters"] = {"test_size": 0.25}
    return CatboostRegressor(config)

def test_timeframe_to_seconds():
    model = get_mock_model()
    assert model.timeframe_to_seconds("5m") == 300
    assert model.timeframe_to_seconds("15m") == 900
    assert model.timeframe_to_seconds("1h") == 3600
    assert model.timeframe_to_seconds("1d") == 86400

def test_apply_purging_and_embargo_chronological():
    model = get_mock_model()
    dk = MagicMock()
    dk.freqai_config = {
        "feature_parameters": {
            "label_period_candles": 3
        },
        "data_split_parameters": {
            "embargo_candles": 5
        }
    }
    dk.config = {
        "timeframe": "5m"
    }

    # Timeframe is 5m (300 seconds)
    # L = 3 * 300 = 900s
    # E = 5 * 300 = 1500s
    # Purging range around a test point t is [t - 900, t + 900 + 1500] = [t - 900, t + 2400]

    # Create synthetic chronological timestamps
    # Test point at 2026-03-01 12:00:00 (Unix: 1772366400)
    test_dt = pd.to_datetime("2026-03-01 12:00:00")
    test_ts = int(test_dt.timestamp())

    # We will generate training points:
    # 1. 2026-03-01 11:40:00 (Unix: 1772365200) -> Diff: -1200s (Outside [test_ts - 900, test_ts + 2400]) -> KEEP
    # 2. 2026-03-01 11:45:00 (Unix: 1772365500) -> Diff: -900s  (Boundary: Inside [test_ts - 900, test_ts + 2400]) -> PURGE
    # 3. 2026-03-01 11:50:00 (Unix: 1772365800) -> Diff: -600s  (Inside) -> PURGE
    # 4. 2026-03-01 12:05:00 (Unix: 1772366700) -> Diff: +300s  (Inside) -> PURGE
    # 5. 2026-03-01 12:40:00 (Unix: 1772368800) -> Diff: +2400s (Boundary: Inside [test_ts - 900, test_ts + 2400]) -> PURGE
    # 6. 2026-03-01 12:45:00 (Unix: 1772369100) -> Diff: +2700s (Outside) -> KEEP

    train_dates_list = [
        "2026-03-01 11:40:00",
        "2026-03-01 11:45:00",
        "2026-03-01 11:50:00",
        "2026-03-01 12:05:00",
        "2026-03-01 12:40:00",
        "2026-03-01 12:45:00",
    ]
    train_dates = pd.to_datetime(train_dates_list)

    # 6 training features
    train_features = pd.DataFrame({"feat": [10, 20, 30, 40, 50, 60]}, index=[0, 1, 2, 3, 4, 5])
    train_labels = pd.DataFrame({"label": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=[0, 1, 2, 3, 4, 5])
    train_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # 1 test feature
    test_features = pd.DataFrame({"feat": [100]}, index=[6])

    # Combined train_dates Series matching all indices
    all_dates = pd.Series(
        data=train_dates_list + ["2026-03-01 12:00:00"],
        index=[0, 1, 2, 3, 4, 5, 6]
    )

    data_dictionary = {
        "train_features": train_features,
        "train_labels": train_labels,
        "train_weights": train_weights,
        "train_dates": all_dates,
        "test_features": test_features
    }

    # Apply purging and embargoing
    X_filtered, y_filtered, weights_filtered = model.apply_purging_and_embargo(data_dictionary, dk)

    # We expect indices 0 and 5 to be KEPT.
    # Indices 1, 2, 3, 4 should be PURGED/EMBARGOED.
    assert len(X_filtered) == 2
    assert list(X_filtered.index) == [0, 5]
    assert list(X_filtered["feat"]) == [10, 60]
    assert list(y_filtered["label"]) == [1.0, 6.0]
    assert list(weights_filtered) == [1.0, 1.0]

def test_apply_purging_and_embargo_empty_cases():
    model = get_mock_model()
    dk = MagicMock()

    # Empty test features case
    data_dict_empty_test = {
        "train_features": pd.DataFrame({"feat": [10, 20]}, index=[0, 1]),
        "train_labels": pd.DataFrame({"label": [1.0, 2.0]}, index=[0, 1]),
        "train_weights": np.array([1.0, 1.0]),
        "train_dates": pd.Series(["2026-03-01 12:00:00", "2026-03-01 12:05:00"], index=[0, 1]),
        "test_features": pd.DataFrame()
    }
    X, y, w = model.apply_purging_and_embargo(data_dict_empty_test, dk)
    assert len(X) == 2  # No purging since test set is empty

    # Missing train_dates case
    data_dict_missing_dates = {
        "train_features": pd.DataFrame({"feat": [10, 20]}, index=[0, 1]),
        "train_labels": pd.DataFrame({"label": [1.0, 2.0]}, index=[0, 1]),
        "train_weights": np.array([1.0, 1.0]),
        "test_features": pd.DataFrame({"feat": [100]}, index=[2])
    }
    X, y, w = model.apply_purging_and_embargo(data_dict_missing_dates, dk)
    assert len(X) == 2  # No purging since dates are missing
