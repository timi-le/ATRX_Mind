"""
ATRX Feature Engineering Module
==============================

Comprehensive feature computation for FX trading data including:
- Technical indicators (RSI, ATR, Bollinger Bands)
- Volatility features (RV, VoV)
- Statistical features (kurtosis, skewness, autocorrelation)
- Microstructure features (spreads, efficiency, imbalance)

Usage:
    from core.features.compute_features import FeatureComputer
    
    computer = FeatureComputer("config/features.yaml")
    computer.process_directory("data/parquet")
"""

from .compute_features import FeatureComputer

__all__ = ['FeatureComputer']
