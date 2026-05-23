import logging
from typing import Any
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class CatboostRegressor(BaseRegressionModel):
    """
    User created prediction model. The class inherits BaseRegressionModel, which
    means it has full access to all Frequency AI functionality.
    """

    def timeframe_to_seconds(self, tf: str) -> int:
        units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
        return int(tf[:-1]) * units[tf[-1]]

    def apply_purging_and_embargo(self, data_dictionary: dict, dk: FreqaiDataKitchen) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        train_weights = data_dictionary["train_weights"]
        train_dates = data_dictionary.get("train_dates", None)
        test_features = data_dictionary.get("test_features", None)

        if train_dates is None or test_features is None or test_features.empty:
            return X, y, train_weights

        # Extract test dates matching test features indices
        train_dates = train_dates.reset_index(drop=True)
        test_dates = train_dates.loc[test_features.index]
        train_dates_subset = train_dates.loc[X.index]

        # Convert to Unix timestamps in seconds
        train_timestamps = pd.to_datetime(train_dates_subset).astype('int64') // 10**9
        test_timestamps = pd.to_datetime(test_dates).astype('int64') // 10**9

        # Parse config parameters
        label_period_candles = dk.freqai_config.get("feature_parameters", {}).get("label_period_candles", 3)
        timeframe = dk.config.get("timeframe", "5m")
        candle_seconds = self.timeframe_to_seconds(timeframe)

        # Purging & Embargo periods
        L = label_period_candles * candle_seconds
        embargo_candles = dk.freqai_config.get("data_split_parameters", {}).get("embargo_candles", 5)
        E = embargo_candles * candle_seconds

        train_ts_arr = train_timestamps.to_numpy()
        test_ts_arr = test_timestamps.to_numpy()

        # Vectorized purged & embargoed ranges check:
        # train point is in [test_ts - L, test_ts + L + E]
        in_purged_range = (train_ts_arr[:, None] >= test_ts_arr - L) & (train_ts_arr[:, None] <= test_ts_arr + L + E)
        keep_mask = ~in_purged_range.any(axis=1)

        original_len = len(X)
        X_filtered = X[keep_mask]
        y_filtered = y[keep_mask]
        train_weights_filtered = train_weights[keep_mask]

        purged_count = original_len - len(X_filtered)
        logger.info(
            f"Purging & Embargoing: Purged {purged_count} overlapping training points out of {original_len} "
            f"(kept {len(X_filtered)}). Params: L={L}s ({label_period_candles} candles), E={E}s ({embargo_candles} candles)"
        )

        return X_filtered, y_filtered, train_weights_filtered

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        # Apply Purging and Embargoing to prevent data leakage from the evaluation set
        X, y, train_weights = self.apply_purging_and_embargo(data_dictionary, dk)

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
        else:
            eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])

        # Initialize the CatBoost regressor
        model = CatBoostRegressor(
            **self.model_training_parameters,
            allow_writing_files=False,
            random_state=self.freqai_info.get('random_state', 42)
        )

        # Train the model
        model.fit(
            X,
            y,
            eval_set=eval_set,
            sample_weight=train_weights,
            verbose=False
        )

        return model
