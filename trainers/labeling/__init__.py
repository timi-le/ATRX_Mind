"""
ATRX Labeling Module
===================

Advanced labeling system for financial time series using Triple-Barrier method
with volatility-based thresholds and VoV filtering for high-quality ML labels.

Features:
- Triple-Barrier labeling (profit, stop-loss, time-based)
- Volatility-adaptive thresholds (ATR/RV basis)
- VoV-based label filtering for market regime awareness
- Multi-asset support with symbol-aware processing

Usage:
    from trainers.labeling.label_regimes import TripleBarrierLabeler
    
    labeler = TripleBarrierLabeler(
        vol_basis='rv',
        k_up=2.0,
        k_dn=2.0,
        horizon=24
    )
    
    labels = labeler.label_features(df)
"""

from .label_regimes import TripleBarrierLabeler

__all__ = ['TripleBarrierLabeler']
