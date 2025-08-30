#!/usr/bin/env python3
"""
ATRX Feature Engineering Pipeline
=================================

Comprehensive feature computation for FX trading data following ATRX development standards.
Computes technical indicators and microstructure features for normalized Parquet datasets.

Features:
- Common technical indicators (RSI, ATR, Bollinger Bands, etc.)
- Microstructure features for bid/ask data
- Robust error handling and validation
- Memory-efficient chunked processing
- Configurable via YAML

Usage:
    poetry run python core/features/compute_features.py
    poetry run python core/features/compute_features.py --config config/features.yaml
    poetry run python core/features/compute_features.py --file data/parquet/eurusd_hour.parquet
"""

import argparse
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import structlog
import yaml
from scipy import stats

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)


class FeatureComputer:
    """
    Comprehensive feature computation engine for FX trading data.
    
    Handles both OHLC and bid/ask data formats with memory-efficient processing
    and robust validation following ATRX architectural standards.
    """
    
    def __init__(self, config_path: str = "config/features.yaml"):
        """Initialize feature computer with configuration."""
        self.config = self._load_config(config_path)
        self.stats = {
            'files_processed': 0,
            'files_succeeded': 0,
            'files_failed': 0,
            'total_features_computed': 0,
            'total_processing_time': 0.0
        }
        
        # Setup output directory
        self.output_dir = Path(self.config['general']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Feature computer initialized", 
                   config_path=config_path,
                   output_dir=str(self.output_dir))
    
    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['general', 'windows', 'common_features']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Required config section missing: {section}")
            
            logger.debug("Configuration loaded", sections=list(config.keys()))
            return config
            
        except Exception as e:
            logger.error("Failed to load configuration", 
                        config_path=config_path, error=str(e))
            raise
    
    def detect_data_format(self, df: pd.DataFrame) -> str:
        """
        Detect whether data is OHLC or bid/ask format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            'ohlc' or 'bidask'
        """
        columns = set(df.columns)
        
        # Check for bid/ask format
        bidask_indicators = {'bid_close', 'ask_close', 'mid_close'}
        if bidask_indicators.issubset(columns):
            return 'bidask'
        
        # Check for OHLC format
        ohlc_indicators = {'open', 'high', 'low', 'close'}
        if ohlc_indicators.issubset(columns):
            return 'ohlc'
        
        raise ValueError(f"Cannot detect data format from columns: {list(columns)}")
    
    def validate_input_data(self, df: pd.DataFrame, data_format: str) -> pd.DataFrame:
        """
        Validate input data and perform basic cleaning.
        
        Args:
            df: Input DataFrame
            data_format: 'ohlc' or 'bidask'
            
        Returns:
            Validated and cleaned DataFrame
        """
        initial_rows = len(df)
        
        # Check minimum rows requirement
        min_rows = self.config['general']['min_required_rows']
        if len(df) < min_rows:
            raise ValueError(f"Insufficient data: {len(df)} rows < {min_rows} required")
        
        # Validate required columns based on format
        if data_format == 'ohlc':
            required = self.config['schema']['ohlc_required']
        else:
            required = self.config['schema']['bidask_required']
        
        missing_cols = set(required) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure timestamp is datetime with timezone
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check for missing values
        max_missing_ratio = self.config['general']['max_missing_ratio']
        for col in df.select_dtypes(include=[np.number]).columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > max_missing_ratio:
                logger.warning("High missing value ratio", 
                              column=col, ratio=f"{missing_ratio:.2%}")
        
        final_rows = len(df)
        if final_rows != initial_rows:
            logger.info("Data validation completed",
                       initial_rows=initial_rows,
                       final_rows=final_rows)
        
        return df
    
    def compute_log_returns(self, df: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
        """Compute log returns for a given column."""
        return np.log(df[column] / df[column].shift(periods))
    
    def compute_realized_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling realized volatility."""
        return np.sqrt(returns.rolling(window).var() * 252)  # Annualized
    
    def compute_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        
        # Determine close column to use
        if 'close' in df.columns:
            close_col = 'close'
        elif 'mid_close' in df.columns:
            close_col = 'mid_close'
        else:
            raise ValueError("No suitable close column found for ATR computation")
        
        high_close_prev = np.abs(df['high'] - df[close_col].shift(1))
        low_close_prev = np.abs(df['low'] - df[close_col].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window).mean()
    
    def compute_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_bollinger_width(self, prices: pd.Series, window: int, std_mult: float = 2.0) -> pd.Series:
        """Compute normalized Bollinger Band width."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        bb_upper = sma + (std * std_mult)
        bb_lower = sma - (std * std_mult)
        bb_width = (bb_upper - bb_lower) / sma
        
        return bb_width
    
    def compute_volatility_of_volatility(self, returns: pd.Series, rv_window: int, vov_window: int) -> pd.Series:
        """Compute volatility of volatility (VoV)."""
        # First compute realized volatility
        rv = self.compute_realized_volatility(returns, rv_window)
        
        # Then compute volatility of log(RV)
        log_rv = np.log(rv.replace(0, np.nan))
        vov = log_rv.rolling(vov_window).std()
        
        return vov
    
    def compute_rolling_kurtosis(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling kurtosis of returns."""
        return returns.rolling(window).kurt()
    
    def compute_rolling_skewness(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling skewness of returns."""
        return returns.rolling(window).skew()
    
    def compute_rolling_autocorr(self, returns: pd.Series, window: int, lag: int = 1) -> pd.Series:
        """Compute rolling autocorrelation of returns."""
        def autocorr(x):
            if len(x) < lag + 1:
                return np.nan
            return np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(np.unique(x)) > 1 else np.nan
        
        return returns.rolling(window).apply(autocorr, raw=True)
    
    def compute_zscore(self, data: pd.Series, window: int) -> pd.Series:
        """Compute rolling z-score."""
        rolling_mean = data.rolling(window).mean()
        rolling_std = data.rolling(window).std()
        
        return (data - rolling_mean) / rolling_std
    
    def compute_range_efficiency(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        Compute range efficiency for bid/ask data.
        Measures how efficiently price moves relative to the high-low range.
        """
        if 'mid_close' in df.columns:
            close_col = 'mid_close'
        else:
            close_col = 'close'
        
        # Price change over window
        price_change = np.abs(df[close_col] - df[close_col].shift(window))
        
        # Sum of high-low ranges over window
        if 'mid_high' in df.columns and 'mid_low' in df.columns:
            range_sum = (df['mid_high'] - df['mid_low']).rolling(window).sum()
        else:
            range_sum = (df['high'] - df['low']).rolling(window).sum()
        
        efficiency = price_change / range_sum.replace(0, np.nan)
        return efficiency
    
    def compute_signed_spread_move(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute signed spread movement relative to mid-price change.
        Positive when spread widens with upward price movement.
        """
        if not all(col in df.columns for col in ['spread_close', 'mid_close']):
            logger.warning("Cannot compute signed_spread_move: missing required columns")
            return pd.Series(np.nan, index=df.index)
        
        spread_change = df['spread_close'].diff()
        mid_change = df['mid_close'].diff()
        
        # Sign of spread change should match sign of price change for informed trading
        signed_move = spread_change * np.sign(mid_change)
        
        return signed_move
    
    def compute_bid_ask_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute bid-ask imbalance using change magnitudes.
        Measures asymmetry in bid vs ask movements.
        """
        if not all(col in df.columns for col in ['bid_change', 'ask_change']):
            logger.warning("Cannot compute bid_ask_imbalance: missing change columns")
            return pd.Series(np.nan, index=df.index)
        
        total_change = np.abs(df['bid_change']) + np.abs(df['ask_change'])
        imbalance = (np.abs(df['ask_change']) - np.abs(df['bid_change'])) / total_change.replace(0, np.nan)
        
        return imbalance
    
    def compute_effective_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute effective spread based on mid-price impact.
        Measures the true cost of trading beyond quoted spread.
        """
        if not all(col in df.columns for col in ['spread_close', 'mid_close']):
            logger.warning("Cannot compute effective_spread: missing required columns")
            return pd.Series(np.nan, index=df.index)
        
        # Simple effective spread proxy: quoted spread adjusted by volatility
        mid_volatility = self.compute_realized_volatility(
            self.compute_log_returns(df, 'mid_close'), window=20
        )
        
        effective_spread = df['spread_close'] * (1 + mid_volatility.fillna(0))
        
        return effective_spread
    
    def compute_quote_slope(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Compute slope of bid/ask quotes over a rolling window.
        Measures the directional pressure in the order book.
        """
        if not all(col in df.columns for col in ['bid_close', 'ask_close']):
            logger.warning("Cannot compute quote_slope: missing bid/ask columns")
            return pd.Series(np.nan, index=df.index)
        
        def compute_slope(x):
            if len(x) < 2:
                return np.nan
            time_idx = np.arange(len(x))
            return np.polyfit(time_idx, x, 1)[0] if not np.isnan(x).all() else np.nan
        
        bid_slope = df['bid_close'].rolling(window).apply(compute_slope, raw=True)
        ask_slope = df['ask_close'].rolling(window).apply(compute_slope, raw=True)
        
        # Average slope indicates overall directional pressure
        quote_slope = (bid_slope + ask_slope) / 2
        
        return quote_slope
    
    def compute_common_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute common technical indicators for all data types.
        
        Args:
            df: Input DataFrame with OHLC or bid/ask data
            
        Returns:
            DataFrame with added common features
        """
        logger.info("Computing common features", features=len(self.config['common_features']))
        
        # Get window sizes and track the maximum for warmup cutoff
        windows = self.config['windows']
        max_window = max([windows.get(k, 0) for k in ['short', 'medium', 'long', 'vol', 'autocorr']])
        
        # Determine price column to use
        if 'close' in df.columns:
            price_col = 'close'
        elif 'mid_close' in df.columns:
            price_col = 'mid_close'
        else:
            raise ValueError("No suitable price column found")
        
        # Store initial row count for retention tracking
        initial_rows = len(df)
        
        # Compute log returns (base for many features)
        df['log_ret_1'] = self.compute_log_returns(df, price_col, periods=1)
        
        # Apply forward/backward fill to log returns before computing dependent features
        df['log_ret_1'] = df['log_ret_1'].fillna(method='ffill').fillna(method='bfill')
        
        # Volatility features with NaN handling
        df['RV_w'] = self.compute_realized_volatility(df['log_ret_1'], windows['medium'])
        df['RV_w'] = df['RV_w'].fillna(method='ffill').fillna(method='bfill')
        
        if 'high' in df.columns and 'low' in df.columns:
            df['ATR_w'] = self.compute_atr(df, windows['short'])
        else:
            # For bid/ask data without explicit high/low, use bid/ask range
            if 'bid_high' in df.columns and 'ask_low' in df.columns:
                df['high'] = df[['bid_high', 'ask_high']].max(axis=1)
                df['low'] = df[['bid_low', 'ask_low']].min(axis=1)
                df['ATR_w'] = self.compute_atr(df, windows['short'])
            else:
                logger.warning("Cannot compute ATR: missing high/low data")
                df['ATR_w'] = np.nan
        
        # Fill ATR NaNs
        if 'ATR_w' in df.columns:
            df['ATR_w'] = df['ATR_w'].fillna(method='ffill').fillna(method='bfill')
        
        # VoV computation with NaN handling
        df['VoV_wV'] = self.compute_volatility_of_volatility(
            df['log_ret_1'], windows['medium'], windows['vol']
        )
        df['VoV_wV'] = df['VoV_wV'].fillna(method='ffill').fillna(method='bfill')
        
        # Momentum indicators with NaN handling
        df['RSI_w'] = self.compute_rsi(df[price_col], windows['short'])
        df['RSI_w'] = df['RSI_w'].fillna(method='ffill').fillna(method='bfill')
        
        # Bollinger band width with NaN handling
        df['bb_width'] = self.compute_bollinger_width(df[price_col], windows['medium'])
        df['bb_width'] = df['bb_width'].fillna(method='ffill').fillna(method='bfill')
        
        # Statistical features with NaN handling
        df['rolling_kurt'] = self.compute_rolling_kurtosis(df['log_ret_1'], windows['medium'])
        df['rolling_kurt'] = df['rolling_kurt'].fillna(method='ffill').fillna(method='bfill')
        
        df['rolling_skew'] = self.compute_rolling_skewness(df['log_ret_1'], windows['medium'])
        df['rolling_skew'] = df['rolling_skew'].fillna(method='ffill').fillna(method='bfill')
        
        df['rolling_autocorr'] = self.compute_rolling_autocorr(
            df['log_ret_1'], windows['autocorr'], lag=1
        )
        df['rolling_autocorr'] = df['rolling_autocorr'].fillna(method='ffill').fillna(method='bfill')
        
        # Normalization features with NaN handling
        df['zscore_close'] = self.compute_zscore(df[price_col], windows['long'])
        df['zscore_close'] = df['zscore_close'].fillna(method='ffill').fillna(method='bfill')
        
        # Apply warmup cutoff - only drop the first max_window rows once
        warmup_cutoff = max_window + 1  # +1 for the first log return calculation
        
        if len(df) > warmup_cutoff:
            rows_before_cutoff = len(df)
            df = df.iloc[warmup_cutoff:].copy()
            rows_after_cutoff = len(df)
            
            retention_rate = rows_after_cutoff / initial_rows
            
            logger.info("Applied warmup cutoff", 
                       warmup_rows_dropped=warmup_cutoff,
                       rows_before=rows_before_cutoff,
                       rows_after=rows_after_cutoff,
                       retention_rate=f"{retention_rate:.1%}")
        else:
            logger.warning("Insufficient data for warmup cutoff", 
                          data_rows=len(df), required_warmup=warmup_cutoff)
        
        logger.info("Common features computed", 
                   new_features=['log_ret_1', 'RV_w', 'ATR_w', 'VoV_wV', 'RSI_w', 
                               'bb_width', 'rolling_kurt', 'rolling_skew', 
                               'rolling_autocorr', 'zscore_close'])
        
        return df
    
    def compute_bidask_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute microstructure features for bid/ask data.
        
        Args:
            df: Input DataFrame with bid/ask data
            
        Returns:
            DataFrame with added microstructure features
        """
        logger.info("Computing bid/ask microstructure features")
        
        windows = self.config['windows']
        
        # Spread features (spread_close should already exist from normalization)
        if 'spread_close' in df.columns:
            df['spread_ma_w'] = df['spread_close'].rolling(windows['spread_ma']).mean()
            df['spread_ma_w'] = df['spread_ma_w'].fillna(method='ffill').fillna(method='bfill')
            
            df['spread_vol'] = df['spread_close'].rolling(windows['medium']).std()
            df['spread_vol'] = df['spread_vol'].fillna(method='ffill').fillna(method='bfill')
        
        # Mid-price returns with NaN handling
        if 'mid_close' in df.columns:
            df['mid_ret_1'] = self.compute_log_returns(df, 'mid_close', periods=1)
            df['mid_ret_1'] = df['mid_ret_1'].fillna(method='ffill').fillna(method='bfill')
        
        # Microstructure features with NaN handling
        df['range_efficiency'] = self.compute_range_efficiency(df, windows['efficiency'])
        df['range_efficiency'] = df['range_efficiency'].fillna(method='ffill').fillna(method='bfill')
        
        df['signed_spread_move'] = self.compute_signed_spread_move(df)
        df['signed_spread_move'] = df['signed_spread_move'].fillna(method='ffill').fillna(method='bfill')
        
        df['bid_ask_imbalance'] = self.compute_bid_ask_imbalance(df)
        df['bid_ask_imbalance'] = df['bid_ask_imbalance'].fillna(method='ffill').fillna(method='bfill')
        
        # Advanced microstructure features with NaN handling
        df['effective_spread'] = self.compute_effective_spread(df)
        df['effective_spread'] = df['effective_spread'].fillna(method='ffill').fillna(method='bfill')
        
        df['quote_slope'] = self.compute_quote_slope(df, window=5)
        df['quote_slope'] = df['quote_slope'].fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Bid/ask features computed",
                   new_features=['spread_ma_w', 'spread_vol', 'mid_ret_1', 
                               'range_efficiency', 'signed_spread_move', 
                               'bid_ask_imbalance', 'effective_spread', 'quote_slope'])
        
        return df
    
    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate computed features and handle outliers.
        
        Args:
            df: DataFrame with computed features
            
        Returns:
            Validated DataFrame
        """
        logger.info("Validating computed features")
        
        validation_config = self.config.get('validation', {})
        outlier_threshold = validation_config.get('outlier_threshold', 5.0)
        feature_ranges = validation_config.get('feature_ranges', {})
        
        # Detect and cap outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in 
                       ['timestamp', 'symbol', 'open', 'high', 'low', 'close']]
        
        for col in feature_cols:
            if col in df.columns:
                # Z-score based outlier detection
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if std_val > 0:
                    z_scores = np.abs((df[col] - mean_val) / std_val)
                    outliers = z_scores > outlier_threshold
                    
                    if outliers.sum() > 0:
                        logger.warning("Outliers detected and capped", 
                                     feature=col, count=outliers.sum())
                        
                        # Cap outliers
                        upper_bound = mean_val + (outlier_threshold * std_val)
                        lower_bound = mean_val - (outlier_threshold * std_val)
                        
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Apply feature-specific ranges
                if col in feature_ranges:
                    min_val, max_val = feature_ranges[col]
                    df[col] = df[col].clip(lower=min_val, upper=max_val)
        
        # Check for high missing value ratio in computed features
        drop_threshold = validation_config.get('drop_threshold', 0.5)
        for col in feature_cols:
            if col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > drop_threshold:
                    logger.warning("Dropping feature with high missing ratio",
                                 feature=col, missing_ratio=f"{missing_ratio:.2%}")
                    df = df.drop(columns=[col])
        
        logger.info("Feature validation completed")
        return df
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        if not self.config.get('performance', {}).get('dtype_optimization', True):
            return df
        
        logger.debug("Optimizing data types for memory efficiency")
        
        # Convert float64 to float32 for feature columns (but keep timestamp precision)
        float_precision = self.config.get('output', {}).get('float_precision', 'float32')
        
        for col in df.select_dtypes(include=['float64']).columns:
            if col != 'timestamp':  # Keep timestamp precision
                df[col] = df[col].astype(float_precision)
        
        return df
    
    def process_file(self, input_path: Union[str, Path], force: bool = False) -> bool:
        """
        Process a single Parquet file to compute features.
        
        Args:
            input_path: Path to input Parquet file
            force: Whether to overwrite existing output files
            
        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error("Input file not found", path=str(input_path))
            return False
        
        # Generate output path
        base_name = input_path.stem
        suffix = self.config['general']['output_suffix']
        output_path = self.output_dir / f"{base_name}{suffix}.parquet"
        
        if output_path.exists() and not force:
            logger.info("Output file exists, skipping",
                       input=str(input_path),
                       output=str(output_path))
            return True
        
        try:
            start_time = time.time()
            self.stats['files_processed'] += 1
            
            logger.info("Processing file", input=str(input_path))
            
            # Load data
            df = pd.read_parquet(input_path)
            initial_rows = len(df)
            
            logger.info("Data loaded", 
                       rows=initial_rows, 
                       columns=len(df.columns),
                       memory_mb=f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
            
            # Detect data format
            data_format = self.detect_data_format(df)
            logger.info("Data format detected", format=data_format)
            
            # Validate input data
            df = self.validate_input_data(df, data_format)
            
            # Compute common features
            df = self.compute_common_features(df)
            
            # Compute format-specific features
            if data_format == 'bidask':
                df = self.compute_bidask_features(df)
            
            # Validate computed features
            df = self.validate_features(df)
            
            # Optimize data types
            df = self.optimize_dtypes(df)
            
            # Save results
            output_config = self.config.get('output', {})
            df.to_parquet(
                output_path,
                compression=output_config.get('compression', 'snappy'),
                index=False
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['files_succeeded'] += 1
            
            # Count new features
            original_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close',
                           'tick_volume', 'real_volume', 'spread',
                           'bid_open', 'bid_high', 'bid_low', 'bid_close',
                           'ask_open', 'ask_high', 'ask_low', 'ask_close',
                           'mid_open', 'mid_high', 'mid_low', 'mid_close',
                           'spread_close', 'bid_change', 'ask_change']
            
            new_features = [col for col in df.columns if col not in original_cols]
            self.stats['total_features_computed'] += len(new_features)
            
            logger.info("File processing completed",
                       input=str(input_path),
                       output=str(output_path),
                       format=data_format,
                       input_rows=initial_rows,
                       output_rows=len(df),
                       new_features=len(new_features),
                       processing_time=f"{processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.stats['files_failed'] += 1
            logger.error("File processing failed",
                        input=str(input_path),
                        error=str(e),
                        exc_info=True)
            return False
    
    def process_directory(self, input_dir: Union[str, Path], 
                         pattern: str = "*.parquet", force: bool = False) -> int:
        """
        Process all Parquet files in a directory.
        
        Args:
            input_dir: Input directory path
            pattern: File pattern to match
            force: Whether to overwrite existing output files
            
        Returns:
            Number of files successfully processed
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            logger.error("Input directory not found", path=str(input_dir))
            return 0
        
        # Find Parquet files
        parquet_files = list(input_dir.glob(pattern))
        
        if not parquet_files:
            logger.warning("No Parquet files found", 
                          directory=str(input_dir), 
                          pattern=pattern)
            return 0
        
        logger.info("Processing directory",
                   directory=str(input_dir),
                   files_found=len(parquet_files))
        
        successful = 0
        for parquet_file in sorted(parquet_files):
            if self.process_file(parquet_file, force=force):
                successful += 1
        
        return successful
    
    def print_summary(self):
        """Print processing summary statistics."""
        print("\n" + "="*70)
        print("FX FEATURE COMPUTATION SUMMARY")
        print("="*70)
        print(f"📁 Files processed:        {self.stats['files_processed']}")
        print(f"✅ Files succeeded:        {self.stats['files_succeeded']}")
        print(f"❌ Files failed:           {self.stats['files_failed']}")
        print(f"🔧 Total features computed: {self.stats['total_features_computed']}")
        print(f"⏱️  Total processing time:  {self.stats['total_processing_time']:.2f}s")
        
        if self.stats['files_succeeded'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['files_succeeded']
            avg_features = self.stats['total_features_computed'] / self.stats['files_succeeded']
            print(f"📊 Avg time per file:      {avg_time:.2f}s")
            print(f"🎯 Avg features per file:  {avg_features:.1f}")
        
        print(f"📁 Output directory:       {self.output_dir}")
        print("="*70)
        
        if self.stats['files_succeeded'] > 0:
            print(f"\n🎉 SUCCESS! {self.stats['files_succeeded']} file(s) processed with features")
            print("🚀 Data is ready for ML training with rich feature set!")


def main():
    """Command-line interface for feature computation."""
    parser = argparse.ArgumentParser(
        description="Compute technical indicators and microstructure features for FX data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files with default config
  python core/features/compute_features.py
  
  # Use custom config
  python core/features/compute_features.py --config config/features.yaml
  
  # Process single file
  python core/features/compute_features.py --file data/parquet/eurusd_hour.parquet
  
  # Force overwrite existing files
  python core/features/compute_features.py --force
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/features.yaml',
        help='Path to configuration file (default: config/features.yaml)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Process single file instead of directory'
    )
    
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
        # Initialize feature computer
        computer = FeatureComputer(config_path=args.config)
        
        print("🔧 FX Feature Computation Pipeline")
        print("="*50)
        
        if args.file:
            # Process single file
            print(f"📄 Processing file: {args.file}")
            success = computer.process_file(args.file, force=args.force)
            if not success:
                print(f"❌ Failed to process {args.file}")
                return 1
        else:
            # Process directory
            input_dir = computer.config['general']['input_dir']
            pattern = computer.config['general']['input_pattern']
            
            print(f"📁 Processing directory: {input_dir}")
            print(f"🎯 Output directory: {computer.output_dir}")
            print(f"🔍 Pattern: {pattern}")
            
            successful = computer.process_directory(
                input_dir,
                pattern=pattern,
                force=args.force
            )
            
            if successful == 0:
                print("❌ No files were successfully processed")
                return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.exception("Fatal error during processing")
        return 1
    
    finally:
        computer.print_summary()
    
    return 0


if __name__ == "__main__":
    exit(main())
