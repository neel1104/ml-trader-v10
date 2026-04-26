import logging
from typing import Any
from catboost import CatBoostRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class CatboostRegressor(BaseRegressionModel):
    """
    User created prediction model. The class inherits BaseRegressionModel, which
    means it has full access to all Frequency AI functionality.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

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
            sample_weight=data_dictionary["train_weights"],
            verbose=False
        )

        return model
