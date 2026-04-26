# tests/test_hyperopt.py
import pytest
import pandas as pd
from datetime import datetime, timedelta

def test_sortino_loss_calculation():
    from user_data.hyperopts.SortinoHyperOptLoss import SortinoHyperOptLoss
    
    loss_class = SortinoHyperOptLoss()
    
    # Mock trade data
    results = pd.DataFrame({
        'profit_ratio': [0.05, -0.02, 0.03, -0.01, 0.04],
        'trade_duration': [10, 5, 15, 8, 12]
    })
    
    trade_count = len(results)
    min_date = datetime.now() - timedelta(days=30)
    max_date = datetime.now()
    
    loss = loss_class.hyperopt_loss_function(
        results,
        trade_count,
        min_date,
        max_date,
        config={}
    )
    
    assert isinstance(loss, float)
    assert loss < 0  # Lower loss is better in Hyperopt
