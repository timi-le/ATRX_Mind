#!/usr/bin/env python3
"""
ATRX Triple-Barrier Regime Labeling System
==========================================

Advanced labeling system for financial time series using the Triple-Barrier method
with volatility-based thresholds and VoV filtering for market regime awareness.

Features:
- Triple-Barrier labeling with profit/stop-loss/time barriers
- Volatility-adaptive thresholds using ATR or RV
- VoV-based filtering for label quality control
- Multi-asset support with symbol-aware processing
- Comprehensive CLI interface for batch processing

Following ATRX development standards for robust, production-ready ML data preparation.

Usage:
    # Via CLI
    python trainers/labeling/label_regimes.py --input data/features/eurusd_hour_features.parquet --vol-basis rv --k-up 2.0 --k-dn 2.0 --horizon 24

    # Via Python API
    from trainers.labeling.label_regimes import TripleBarrierLabeler
    
    labeler = TripleBarrierLabeler(vol_basis='rv', k_up=2.0, k_dn=2.0, horizon=24)
    labels = labeler.label_features(df)
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import structlog

# Optional numba import for performance optimization
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback for systems without numba
    NUMBA_AVAILABLE = False
    def jit(nopython=True):
        """Fallback decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)


@jit(nopython=True)
def _triple_barrier_numba(prices: np.ndarray, upper_barriers: np.ndarray, 
                         lower_barriers: np.ndarray, horizon: int) -> np.ndarray:
    """
    Numba-optimized triple-barrier labeling for performance.
    
    Args:
        prices: Array of price values
        upper_barriers: Array of upper barrier levels
        lower_barriers: Array of lower barrier levels  
        horizon: Maximum holding period in bars
        
    Returns:
        Array of labels: +1 (upper hit), -1 (lower hit), 0 (timeout)
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.float64)
    
    for i in range(n - 1):  # Skip last observation (no future data)
        if np.isnan(upper_barriers[i]) or np.isnan(lower_barriers[i]):
            labels[i] = np.nan
            continue
            
        upper_thresh = upper_barriers[i]
        lower_thresh = lower_barriers[i]
        start_price = prices[i]
        
        # Look forward up to horizon bars
        max_lookahead = min(horizon, n - i - 1)
        
        for j in range(1, max_lookahead + 1):
            current_price = prices[i + j]
            
            if np.isnan(current_price):
                continue
                
            # Check if barriers are hit
            if current_price >= upper_thresh:
                labels[i] = 1.0  # Upper barrier hit (profit/long signal)
                break
            elif current_price <= lower_thresh:
                labels[i] = -1.0  # Lower barrier hit (stop-loss/short signal)
                break
        
        # If no barrier hit within horizon, label as timeout (0)
        # labels[i] remains 0.0 from initialization
    
    return labels


class TripleBarrierLabeler:
    """
    Advanced Triple-Barrier labeling system with volatility adaptation and VoV filtering.
    
    The Triple-Barrier method sets profit-taking and stop-loss levels based on
    volatility estimates, then assigns labels based on which barrier is hit first
    within a specified time horizon.
    """
    
    def __init__(self, vol_basis: str = 'rv', price_col: str = 'close',
                 k_up: float = 2.0, k_dn: float = 2.0, horizon: int = 24,
                 vov_col: str = 'VoV_wV', vov_lower_quant: float = 0.2,
                 vov_upper_quant: float = 0.8):
        """
        Initialize the Triple-Barrier labeler.
        
        Args:
            vol_basis: Volatility basis for thresholds ('rv' or 'atr')
            price_col: Price column to use ('close', 'mid_close', etc.)
            k_up: Multiplier for upper barrier (profit taking)
            k_dn: Multiplier for lower barrier (stop loss)
            horizon: Maximum holding period in bars
            vov_col: VoV column name for filtering
            vov_lower_quant: Lower quantile for VoV filtering
            vov_upper_quant: Upper quantile for VoV filtering
        """
        self.vol_basis = vol_basis
        self.price_col = price_col
        self.k_up = k_up
        self.k_dn = k_dn
        self.horizon = horizon
        self.vov_col = vov_col
        self.vov_lower_quant = vov_lower_quant
        self.vov_upper_quant = vov_upper_quant
        
        # Validate parameters
        self._validate_parameters()
        
        # Statistics tracking
        self.stats = {
            'total_observations': 0,
            'valid_labels': 0,
            'upper_hits': 0,
            'lower_hits': 0,
            'timeouts': 0,
            'vov_filtered': 0,
            'processing_time': 0.0
        }
        
        logger.info("Triple-Barrier labeler initialized",
                   vol_basis=vol_basis, price_col=price_col,
                   k_up=k_up, k_dn=k_dn, horizon=horizon,
                   vov_filtering=f"{vov_lower_quant:.1%}-{vov_upper_quant:.1%}",
                   numba_enabled=NUMBA_AVAILABLE)
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if self.vol_basis not in ['rv', 'atr']:
            raise ValueError(f"vol_basis must be 'rv' or 'atr', got '{self.vol_basis}'")
        
        if self.k_up <= 0 or self.k_dn <= 0:
            raise ValueError("k_up and k_dn must be positive")
        
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        
        if not (0 <= self.vov_lower_quant <= self.vov_upper_quant <= 1):
            raise ValueError("VoV quantiles must be between 0 and 1, with lower <= upper")
    
    def _get_volatility_column(self, df: pd.DataFrame) -> str:
        """Determine the volatility column to use based on vol_basis."""
        if self.vol_basis == 'rv':
            vol_col = 'RV_w'
        elif self.vol_basis == 'atr':
            vol_col = 'ATR_w'
        else:
            raise ValueError(f"Unknown vol_basis: {self.vol_basis}")
        
        if vol_col not in df.columns:
            raise ValueError(f"Required volatility column '{vol_col}' not found in data. "
                           f"Available columns: {list(df.columns)}")
        
        return vol_col
    
    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data."""
        # Check required columns
        required_cols = ['timestamp', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check price column
        if self.price_col not in df.columns:
            raise ValueError(f"Price column '{self.price_col}' not found. "
                           f"Available columns: {list(df.columns)}")
        
        # Check volatility column
        vol_col = self._get_volatility_column(df)
        
        # Check VoV column if filtering enabled
        if self.vov_col not in df.columns:
            logger.warning(f"VoV column '{self.vov_col}' not found. "
                          f"VoV filtering will be disabled.")
            self.vov_col = None
        
        # Ensure data is sorted by timestamp
        if not df['timestamp'].is_monotonic_increasing:
            logger.info("Sorting data by timestamp")
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove rows with missing price or volatility data
        initial_rows = len(df)
        df = df.dropna(subset=[self.price_col, vol_col])
        final_rows = len(df)
        
        if final_rows < initial_rows:
            logger.info("Removed rows with missing data",
                       initial=initial_rows, final=final_rows,
                       removed=initial_rows - final_rows)
        
        if final_rows < self.horizon:
            raise ValueError(f"Insufficient data: {final_rows} rows < {self.horizon} horizon")
        
        return df
    
    def _compute_vov_thresholds(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Compute VoV filtering thresholds based on quantiles."""
        if self.vov_col is None or self.vov_col not in df.columns:
            return -np.inf, np.inf
        
        vov_values = df[self.vov_col].dropna()
        
        if len(vov_values) == 0:
            logger.warning("No valid VoV values found, disabling VoV filtering")
            return -np.inf, np.inf
        
        lower_thresh = vov_values.quantile(self.vov_lower_quant)
        upper_thresh = vov_values.quantile(self.vov_upper_quant)
        
        logger.info("VoV thresholds computed",
                   lower_thresh=f"{lower_thresh:.6f}",
                   upper_thresh=f"{upper_thresh:.6f}",
                   lower_quant=f"{self.vov_lower_quant:.1%}",
                   upper_quant=f"{self.vov_upper_quant:.1%}")
        
        return lower_thresh, upper_thresh
    
    def _apply_vov_filtering(self, df: pd.DataFrame, labels: np.ndarray,
                           vov_lower: float, vov_upper: float) -> np.ndarray:
        """Apply VoV-based filtering to remove labels in extreme volatility regimes."""
        if self.vov_col is None or self.vov_col not in df.columns:
            return labels
        
        vov_values = df[self.vov_col].values
        
        # Create mask for VoV values outside acceptable range
        vov_filter = (vov_values < vov_lower) | (vov_values > vov_upper)
        
        # Apply filter by setting labels to NaN
        filtered_labels = labels.copy()
        filtered_labels[vov_filter] = np.nan
        
        filtered_count = np.sum(vov_filter)
        self.stats['vov_filtered'] = filtered_count
        
        logger.info("VoV filtering applied",
                   filtered_observations=filtered_count,
                   percentage=f"{filtered_count/len(labels)*100:.1f}%")
        
        return filtered_labels
    
    def _compute_barriers(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Compute upper and lower barrier levels based on volatility."""
        vol_col = self._get_volatility_column(df)
        
        prices = df[self.price_col].values
        volatility = df[vol_col].values
        
        # Compute barriers: price ± k * volatility
        upper_barriers = prices + (self.k_up * volatility)
        lower_barriers = prices - (self.k_dn * volatility)
        
        logger.debug("Barriers computed",
                    price_col=self.price_col, vol_col=vol_col,
                    mean_upper_spread=f"{np.nanmean(upper_barriers - prices):.6f}",
                    mean_lower_spread=f"{np.nanmean(prices - lower_barriers):.6f}")
        
        return upper_barriers, lower_barriers
    
    def label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Triple-Barrier labeling to the input data.
        
        Args:
            df: Input DataFrame with OHLC/price data and features
            
        Returns:
            DataFrame with timestamp, symbol, and label columns
        """
        start_time = time.time()
        
        # Validate and prepare data
        df = self._validate_input_data(df)
        self.stats['total_observations'] = len(df)
        
        logger.info("Starting Triple-Barrier labeling",
                   observations=len(df),
                   symbols=df['symbol'].nunique(),
                   vol_basis=self.vol_basis,
                   horizon=self.horizon)
        
        # Compute VoV thresholds for filtering
        vov_lower, vov_upper = self._compute_vov_thresholds(df)
        
        # Process each symbol separately to handle regime differences
        symbol_results = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            logger.debug("Processing symbol", symbol=symbol, rows=len(symbol_df))
            
            # Compute barriers
            upper_barriers, lower_barriers = self._compute_barriers(symbol_df)
            
            # Apply triple-barrier labeling using optimized Numba function
            prices = symbol_df[self.price_col].values
            labels = _triple_barrier_numba(prices, upper_barriers, lower_barriers, self.horizon)
            
            # Apply VoV filtering
            labels = self._apply_vov_filtering(symbol_df, labels, vov_lower, vov_upper)
            
            # Create result DataFrame for this symbol
            symbol_result = pd.DataFrame({
                'timestamp': symbol_df['timestamp'],
                'symbol': symbol_df['symbol'],
                'label': labels
            })
            
            symbol_results.append(symbol_result)
            
            # Update statistics
            valid_mask = ~np.isnan(labels)
            symbol_valid = np.sum(valid_mask)
            symbol_upper = np.sum(labels[valid_mask] == 1)
            symbol_lower = np.sum(labels[valid_mask] == -1)
            symbol_timeout = np.sum(labels[valid_mask] == 0)
            
            logger.debug("Symbol labeling completed",
                        symbol=symbol,
                        valid_labels=symbol_valid,
                        upper_hits=symbol_upper,
                        lower_hits=symbol_lower,
                        timeouts=symbol_timeout)
        
        # Combine results from all symbols
        result_df = pd.concat(symbol_results, ignore_index=True)
        
        # Update final statistics
        valid_labels = result_df['label'].notna()
        self.stats['valid_labels'] = valid_labels.sum()
        self.stats['upper_hits'] = (result_df['label'] == 1).sum()
        self.stats['lower_hits'] = (result_df['label'] == -1).sum()
        self.stats['timeouts'] = (result_df['label'] == 0).sum()
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info("Triple-Barrier labeling completed",
                   total_observations=self.stats['total_observations'],
                   valid_labels=self.stats['valid_labels'],
                   upper_hits=self.stats['upper_hits'],
                   lower_hits=self.stats['lower_hits'],
                   timeouts=self.stats['timeouts'],
                   vov_filtered=self.stats['vov_filtered'],
                   processing_time=f"{self.stats['processing_time']:.2f}s")
        
        return result_df
    
    def print_statistics(self):
        """Print detailed labeling statistics."""
        print("\n" + "="*70)
        print("TRIPLE-BARRIER LABELING STATISTICS")
        print("="*70)
        print(f"📊 Total observations:     {self.stats['total_observations']:,}")
        print(f"✅ Valid labels:           {self.stats['valid_labels']:,} "
              f"({self.stats['valid_labels']/max(self.stats['total_observations'], 1)*100:.1f}%)")
        print(f"🔼 Upper hits (+1):        {self.stats['upper_hits']:,} "
              f"({self.stats['upper_hits']/max(self.stats['valid_labels'], 1)*100:.1f}%)")
        print(f"🔽 Lower hits (-1):        {self.stats['lower_hits']:,} "
              f"({self.stats['lower_hits']/max(self.stats['valid_labels'], 1)*100:.1f}%)")
        print(f"⏱️  Timeouts (0):          {self.stats['timeouts']:,} "
              f"({self.stats['timeouts']/max(self.stats['valid_labels'], 1)*100:.1f}%)")
        print(f"🔽 VoV filtered:           {self.stats['vov_filtered']:,} "
              f"({self.stats['vov_filtered']/max(self.stats['total_observations'], 1)*100:.1f}%)")
        print(f"⏱️  Processing time:       {self.stats['processing_time']:.2f}s")
        
        # Label balance analysis
        if self.stats['valid_labels'] > 0:
            upper_pct = self.stats['upper_hits'] / self.stats['valid_labels'] * 100
            lower_pct = self.stats['lower_hits'] / self.stats['valid_labels'] * 100
            timeout_pct = self.stats['timeouts'] / self.stats['valid_labels'] * 100
            
            print(f"\n📈 Label Distribution:")
            print(f"   Upper hits:  {upper_pct:5.1f}%")
            print(f"   Lower hits:  {lower_pct:5.1f}%") 
            print(f"   Timeouts:    {timeout_pct:5.1f}%")
            
            # Balance assessment
            imbalance = abs(upper_pct - lower_pct)
            if imbalance < 10:
                print(f"✅ Labels are well balanced (imbalance: {imbalance:.1f}%)")
            elif imbalance < 20:
                print(f"⚠️  Moderate label imbalance ({imbalance:.1f}%)")
            else:
                print(f"❌ Significant label imbalance ({imbalance:.1f}%)")
        
        print("="*70)


def load_feature_data(input_path: Union[str, Path]) -> pd.DataFrame:
    """Load feature data from Parquet file."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.suffix.lower() != '.parquet':
        raise ValueError(f"Input file must be a Parquet file, got: {input_path.suffix}")
    
    try:
        df = pd.read_parquet(input_path)
        logger.info("Feature data loaded",
                   file=str(input_path),
                   rows=len(df),
                   columns=len(df.columns),
                   memory_mb=f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load feature data from {input_path}: {e}")


def save_labels(labels_df: pd.DataFrame, output_path: Union[str, Path]):
    """Save labels to Parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        labels_df.to_parquet(output_path, compression='snappy', index=False)
        logger.info("Labels saved",
                   file=str(output_path),
                   rows=len(labels_df),
                   memory_mb=f"{labels_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    except Exception as e:
        raise RuntimeError(f"Failed to save labels to {output_path}: {e}")


def main():
    """Command-line interface for Triple-Barrier labeling."""
    parser = argparse.ArgumentParser(
        description="Generate Triple-Barrier labels with VoV filtering for FX data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic labeling with RV basis
  python trainers/labeling/label_regimes.py --input data/features/eurusd_hour_features.parquet

  # Custom parameters with ATR basis
  python trainers/labeling/label_regimes.py \\
    --input data/features/eurusd_hour_features.parquet \\
    --vol-basis atr --k-up 2.5 --k-dn 2.0 --horizon 48

  # VoV filtering for regime awareness
  python trainers/labeling/label_regimes.py \\
    --input data/features/eurusd_hour_features.parquet \\
    --vov-lower-quant 0.1 --vov-upper-quant 0.9

  # Multiple files batch processing
  python trainers/labeling/label_regimes.py \\
    --input "data/features/*_features.parquet" \\
    --output-dir data/labels/
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input Parquet file with features (supports glob patterns)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output Parquet file for labels (default: auto-generated from input)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/labels',
        help='Output directory for labels (default: data/labels)'
    )
    
    # Labeling parameters
    parser.add_argument(
        '--vol-basis',
        choices=['rv', 'atr'],
        default='rv',
        help='Volatility basis for barriers (default: rv)'
    )
    
    parser.add_argument(
        '--price-col',
        type=str,
        default='close',
        help='Price column name (default: close)'
    )
    
    parser.add_argument(
        '--k-up',
        type=float,
        default=2.0,
        help='Upper barrier multiplier (default: 2.0)'
    )
    
    parser.add_argument(
        '--k-dn',
        type=float,
        default=2.0,
        help='Lower barrier multiplier (default: 2.0)'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        default=24,
        help='Maximum holding period in bars (default: 24)'
    )
    
    # VoV filtering parameters
    parser.add_argument(
        '--vov-col',
        type=str,
        default='VoV_wV',
        help='VoV column name for filtering (default: VoV_wV)'
    )
    
    parser.add_argument(
        '--vov-lower-quant',
        type=float,
        default=0.2,
        help='Lower VoV quantile threshold (default: 0.2)'
    )
    
    parser.add_argument(
        '--vov-upper-quant',
        type=float,
        default=0.8,
        help='Upper VoV quantile threshold (default: 0.8)'
    )
    
    # Processing options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print("🏷️  ATRX Triple-Barrier Labeling System")
        print("="*50)
        
        # Initialize labeler
        labeler = TripleBarrierLabeler(
            vol_basis=args.vol_basis,
            price_col=args.price_col,
            k_up=args.k_up,
            k_dn=args.k_dn,
            horizon=args.horizon,
            vov_col=args.vov_col,
            vov_lower_quant=args.vov_lower_quant,
            vov_upper_quant=args.vov_upper_quant
        )
        
        # Handle glob patterns for input
        from glob import glob
        input_files = glob(args.input) if '*' in args.input else [args.input]
        
        if not input_files:
            raise FileNotFoundError(f"No files found matching pattern: {args.input}")
        
        print(f"📁 Processing {len(input_files)} file(s)")
        
        for input_file in sorted(input_files):
            print(f"\n📄 Processing: {input_file}")
            
            # Load feature data
            df = load_feature_data(input_file)
            
            # Generate labels
            labels_df = labeler.label_data(df)
            
            # Determine output path
            if args.output:
                output_path = args.output
            else:
                input_path = Path(input_file)
                output_path = Path(args.output_dir) / f"{input_path.stem}_labels.parquet"
            
            # Check if output exists
            if Path(output_path).exists() and not args.force:
                print(f"⚠️  Output file exists: {output_path}")
                print("   Use --force to overwrite")
                continue
            
            # Save labels
            save_labels(labels_df, output_path)
            
            # Print statistics
            labeler.print_statistics()
            
            print(f"✅ Labels saved to: {output_path}")
        
        print(f"\n🎉 Processing completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Fatal error during labeling")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
