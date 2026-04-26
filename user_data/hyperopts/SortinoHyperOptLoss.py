# user_data/hyperopts/SortinoHyperOptLoss.py
import numpy as np
from datetime import datetime
from pandas import DataFrame
from freqtrade.optimize.hyperopt import IHyperOptLoss

class SortinoHyperOptLoss(IHyperOptLoss):
    """
    Defines a custom loss function for Hyperopt that targets the Sortino Ratio.
    The Sortino ratio measures the risk-adjusted return of an investment asset,
    portfolio, or strategy. It is a modification of the Sharpe ratio but penalizes
    only those returns falling below a user-specified target or required rate of return,
    while the Sharpe ratio penalizes both upside and downside volatility equally.
    """
    
    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: dict, *args, **kwargs) -> float:
        """
        Objective function.
        
        Returns the Sortino Ratio calculated from the trades.
        Since Hyperopt minimizes the loss, we return the negative Sortino Ratio.
        """
        if trade_count == 0:
            return 1000.0 # High loss for no trades
            
        profit_ratios = results['profit_ratio']
        
        # Calculate expected return (annualized)
        # Using a simple sum for now, can be refined based on backtest duration
        total_return = profit_ratios.sum()
        
        # Calculate downside deviation
        # Filter returns < 0
        downside_returns = profit_ratios[profit_ratios < 0]
        
        if len(downside_returns) == 0:
            # No losing trades, very good!
            return -total_return * 100 
            
        # Calculate the target downside deviation (standard deviation of negative returns)
        # We use a target return of 0 for simplicity
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
        
        if downside_deviation == 0:
             return -total_return * 100
             
        sortino_ratio = total_return / downside_deviation
        
        # Return negative sortino ratio since hyperopt minimizes the loss function
        return -sortino_ratio
