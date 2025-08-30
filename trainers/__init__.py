"""
Trainers module for FX AI-Quant Trading System.

This module provides production-grade training pipeline components including:
- Cross-validation utilities for time-series data
- Production training pipeline with regularization and early stopping
- Walk-forward optimization and hyperparameter tuning
- Model management and ONNX export capabilities
"""

from .cv_utils import (
    CVResult,
    PurgedKFold,
    TimeSeriesKFold,
    WalkForwardOptimizer,
    calmar_ratio_score,
    information_ratio_score,
    sharpe_ratio_score,
    validate_cv_setup,
)
from .train_model import ProductionTrainingPipeline, TrainingResult

__all__ = [
    # Cross-validation utilities
    "TimeSeriesKFold",
    "WalkForwardOptimizer",
    "PurgedKFold",
    "CVResult",
    "sharpe_ratio_score",
    "information_ratio_score",
    "calmar_ratio_score",
    "validate_cv_setup",
    # Training pipeline
    "ProductionTrainingPipeline",
    "TrainingResult",
]
