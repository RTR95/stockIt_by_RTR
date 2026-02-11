"""
Backtesting module for Time-Series Forecasters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time

from .forecaster import TimeSeriesForecaster
from .base import ForecastOutput

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    coverage_80: float  # Percentage of actuals within 80% CI
    total_samples: int
    predictions: List[Dict[str, Any]]
    metrics_by_horizon: Dict[int, Dict[str, float]]


class ForecasterBacktester:
    """
    Performs walk-forward validation (backtesting) on time-series forecasters.
    """
    
    def __init__(self, forecaster: TimeSeriesForecaster):
        self.forecaster = forecaster

    def run(
        self,
        price_history: pd.Series,
        test_size: int = 180,  # Test on last 180 days
        min_train_size: int = 60,
        step_size: int = 15,
        horizon: int = 30,
        verbose: bool = False
    ) -> BacktestResult:
        """
        Run walk-forward validation.
        
        Args:
            price_history: Full price history (pd.Series with DatetimeIndex)
            test_size: Number of recent days to use for testing
            min_train_size: Minimum history needed for model context
            step_size: How many days to move forward in each step (e.g., re-forecast every 15 days)
            horizon: Forecast horizon
            verbose: Print progress
            
        Returns:
            BacktestResult with aggregate metrics
        """
        if len(price_history) < min_train_size + horizon:
            raise ValueError(f"Insufficient history: {len(price_history)} < {min_train_size} + {horizon}")
            
        # Ensure model is loaded
        if not self.forecaster.is_ready:
            if not self.forecaster.load_model():
                raise RuntimeError("Failed to load forecaster model")

        predictions = []
        errors_abs = []
        errors_sq = []
        errors_pct = []
        hits_direction = []
        hits_coverage = []
        
        # Determine split points
        total_len = len(price_history)
        start_idx = max(min_train_size, total_len - test_size)
        
        # Walk forward
        for i in range(start_idx, total_len - horizon, step_size):
            train_data = price_history.iloc[:i]
            actual_future = price_history.iloc[i : i + horizon]
            
            if len(actual_future) < horizon:
                break
                
            if verbose:
                logger.info(f"Backtesting window: {train_data.index[-1]} -> predicting next {horizon} days")
            
            # Generate forecast
            forecast = self.forecaster.forecast(
                train_data,
                horizon=horizon,
                num_samples=100
            )
            
            if not forecast.success:
                logger.warning(f"Forecast failed at {train_data.index[-1]}: {forecast.error}")
                continue
                
            pred_values = forecast.predictions
            
            # Calculate metrics for this window
            actual_values = actual_future.values
            
            # Absolute Error
            ae = np.abs(pred_values - actual_values)
            errors_abs.extend(ae)
            
            # Squared Error
            se = (pred_values - actual_values) ** 2
            errors_sq.extend(se)
            
            # Percentage Error
            pe = np.abs((pred_values - actual_values) / actual_values)
            errors_pct.extend(pe)
            
            # Directional Accuracy (Trend)
            current_price = train_data.iloc[-1]
            pred_direction = 1 if pred_values[-1] > current_price else -1
            actual_direction = 1 if actual_values[-1] > current_price else -1
            hits_direction.append(1 if pred_direction == actual_direction else 0)
            
            # Coverage (80% CI)
            # forecast.prediction_intervals is a dict
            lower = forecast.prediction_intervals['lower_10']
            upper = forecast.prediction_intervals['upper_90']
            
            covered = np.sum((actual_values >= lower) & (actual_values <= upper))
            hits_coverage.append(covered / horizon)
            
            predictions.append({
                'date': train_data.index[-1],
                'actual': actual_values.tolist(),
                'predicted': pred_values.tolist(),
                'lower_bound': lower.tolist(),
                'upper_bound': upper.tolist(),
                'horizon': horizon
            })
            
        if not errors_abs:
            return BacktestResult(0, 0, 0, 0, 0, 0, [], {})
            
        # Aggregate metrics
        mae = float(np.mean(errors_abs))
        rmse = float(np.sqrt(np.mean(errors_sq)))
        mape = float(np.mean(errors_pct)) * 100
        dir_acc = float(np.mean(hits_direction)) * 100
        cov_80 = float(np.mean(hits_coverage)) * 100
        
        # Metrics by horizon (e.g., Day 1, Day 5, Day 30)
        metrics_by_h = {}
        # This would require reshaping errors to (num_windows, horizon)
        # For simplicity, we skip granular horizon breakdown in this version
        
        return BacktestResult(
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            mape=round(mape, 2),
            directional_accuracy=round(dir_acc, 1),
            coverage_80=round(cov_80, 1),
            total_samples=len(errors_abs),
            predictions=predictions,
            metrics_by_horizon=metrics_by_h
        )
