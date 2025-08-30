#!/usr/bin/env python3
"""
ATRX Time Series Splitting System
=================================

Walk-forward time series splitting for robust ML model validation.
Prevents data leakage and maintains temporal ordering for financial data.

Features:
- Walk-forward splits with configurable train/validation windows
- Chronological ordering with no data leakage
- Gap insertion between train/validation to prevent lookahead bias
- Multi-symbol support with symbol-aware splitting
- Comprehensive validation and coverage reporting
- Flexible output formats (indices, metadata, or data files)

Following ATRX development standards for production-ready ML validation.

Usage:
    # Basic walk-forward splitting
    python scripts/split_time_series.py --input data/datasets/eurusd_hour_dataset.parquet --train-years 2 --val-years 1

    # Advanced splitting with gaps
    python scripts/split_time_series.py --input data/datasets/eurusd_hour_dataset.parquet --train-years 3 --val-years 0.5 --step-years 1 --gap-days 7

    # Multiple datasets
    python scripts/split_time_series.py --input "data/datasets/*_dataset.parquet" --train-years 2 --val-years 1 --output-format data
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import structlog
from glob import glob

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)


class TimeSeriesSplitter:
    """
    Advanced walk-forward time series splitter for financial ML validation.
    
    Ensures proper temporal ordering, prevents data leakage, and provides
    comprehensive validation of split quality.
    """
    
    def __init__(self, train_years: float = 2.0, val_years: float = 1.0,
                 step_years: float = 1.0, gap_days: int = 0,
                 min_samples_per_split: int = 100):
        """
        Initialize the time series splitter.
        
        Args:
            train_years: Training window size in years
            val_years: Validation window size in years  
            step_years: Step size between splits in years
            gap_days: Gap between train and validation in days (prevents leakage)
            min_samples_per_split: Minimum samples required per split
        """
        self.train_years = train_years
        self.val_years = val_years
        self.step_years = step_years
        self.gap_days = gap_days
        self.min_samples_per_split = min_samples_per_split
        
        # Convert to timedelta for calculations
        self.train_delta = timedelta(days=train_years * 365.25)
        self.val_delta = timedelta(days=val_years * 365.25)
        self.step_delta = timedelta(days=step_years * 365.25)
        self.gap_delta = timedelta(days=gap_days)
        
        # Statistics tracking
        self.stats = {
            'total_splits': 0,
            'valid_splits': 0,
            'total_train_samples': 0,
            'total_val_samples': 0,
            'coverage_ratio': 0.0,
            'overlap_ratio': 0.0,
            'processing_time': 0.0
        }
        
        logger.info("Time series splitter initialized",
                   train_years=train_years,
                   val_years=val_years,
                   step_years=step_years,
                   gap_days=gap_days,
                   min_samples=min_samples_per_split)
    
    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input dataset for splitting."""
        # Check required columns
        required_cols = ['timestamp', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logger.info("Converting timestamp to datetime")
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Sort by symbol and timestamp
        if not df['timestamp'].is_monotonic_increasing:
            logger.info("Sorting data by timestamp")
            df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Check minimum data requirements
        if len(df) < self.min_samples_per_split * 2:
            raise ValueError(f"Insufficient data: {len(df)} < {self.min_samples_per_split * 2} required")
        
        # Analyze data coverage
        date_range = df['timestamp'].max() - df['timestamp'].min()
        required_range = self.train_delta + self.val_delta + self.gap_delta
        
        if date_range < required_range:
            logger.warning("Limited time range for splitting",
                          available=f"{date_range.days} days",
                          required=f"{required_range.days} days")
        
        logger.info("Input data validated",
                   rows=len(df),
                   symbols=df['symbol'].nunique(),
                   date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                   span_days=date_range.days)
        
        return df
    
    def _generate_split_windows(self, start_date: pd.Timestamp, 
                               end_date: pd.Timestamp) -> List[Dict]:
        """Generate all possible split windows within the date range."""
        splits = []
        split_id = 0
        
        current_start = start_date
        
        while True:
            # Calculate window boundaries
            train_start = current_start
            train_end = train_start + self.train_delta
            
            # Add gap if specified
            val_start = train_end + self.gap_delta
            val_end = val_start + self.val_delta
            
            # Check if we have enough data for this split
            if val_end > end_date:
                break
            
            split_info = {
                'split_id': split_id,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'gap_days': self.gap_days
            }
            
            splits.append(split_info)
            split_id += 1
            
            # Move to next split
            current_start += self.step_delta
        
        logger.info("Split windows generated",
                   total_splits=len(splits),
                   first_train_start=splits[0]['train_start'] if splits else None,
                   last_val_end=splits[-1]['val_end'] if splits else None)
        
        return splits
    
    def _extract_split_data(self, df: pd.DataFrame, symbol: str, 
                           split_info: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract train and validation data for a specific split."""
        # Filter by symbol
        symbol_data = df[df['symbol'] == symbol].copy()
        
        # Extract training data
        train_mask = (
            (symbol_data['timestamp'] >= split_info['train_start']) &
            (symbol_data['timestamp'] < split_info['train_end'])
        )
        train_data = symbol_data[train_mask].copy()
        
        # Extract validation data
        val_mask = (
            (symbol_data['timestamp'] >= split_info['val_start']) &
            (symbol_data['timestamp'] < split_info['val_end'])
        )
        val_data = symbol_data[val_mask].copy()
        
        return train_data, val_data
    
    def _validate_split_quality(self, train_data: pd.DataFrame, 
                               val_data: pd.DataFrame, split_info: Dict) -> bool:
        """Validate the quality of a single split."""
        # Check minimum sample requirements
        if len(train_data) < self.min_samples_per_split:
            logger.debug("Split rejected: insufficient training samples",
                        split_id=split_info['split_id'],
                        train_samples=len(train_data),
                        required=self.min_samples_per_split)
            return False
        
        if len(val_data) < self.min_samples_per_split:
            logger.debug("Split rejected: insufficient validation samples",
                        split_id=split_info['split_id'],
                        val_samples=len(val_data),
                        required=self.min_samples_per_split)
            return False
        
        # Check temporal ordering (no leakage)
        if not train_data.empty and not val_data.empty:
            max_train_time = train_data['timestamp'].max()
            min_val_time = val_data['timestamp'].min()
            
            if max_train_time >= min_val_time:
                logger.warning("Split rejected: temporal leakage detected",
                              split_id=split_info['split_id'],
                              max_train=max_train_time,
                              min_val=min_val_time)
                return False
        
        # Check for reasonable time gaps
        if self.gap_days > 0 and not train_data.empty and not val_data.empty:
            actual_gap = val_data['timestamp'].min() - train_data['timestamp'].max()
            expected_gap = self.gap_delta
            
            gap_tolerance = timedelta(days=1)  # Allow 1 day tolerance
            if abs(actual_gap - expected_gap) > gap_tolerance:
                logger.debug("Split has unexpected gap",
                           split_id=split_info['split_id'],
                           actual_gap=actual_gap.days,
                           expected_gap=expected_gap.days)
        
        return True
    
    def _analyze_coverage_and_overlap(self, splits: List[Dict], df: pd.DataFrame) -> Dict:
        """Analyze data coverage and overlap across splits."""
        total_data_points = len(df)
        covered_indices = set()
        overlap_count = 0
        
        for split in splits:
            # Calculate indices covered by this split's train and val data
            train_mask = (
                (df['timestamp'] >= split['train_start']) &
                (df['timestamp'] < split['train_end'])
            )
            val_mask = (
                (df['timestamp'] >= split['val_start']) &
                (df['timestamp'] < split['val_end'])
            )
            
            train_indices = set(df[train_mask].index)
            val_indices = set(df[val_mask].index)
            split_indices = train_indices.union(val_indices)
            
            # Check for overlap with previously covered data
            overlap = len(covered_indices.intersection(split_indices))
            overlap_count += overlap
            
            # Add to covered indices
            covered_indices.update(split_indices)
        
        coverage_ratio = len(covered_indices) / total_data_points
        overlap_ratio = overlap_count / total_data_points if total_data_points > 0 else 0
        
        return {
            'coverage_ratio': coverage_ratio,
            'overlap_ratio': overlap_ratio,
            'covered_samples': len(covered_indices),
            'total_samples': total_data_points
        }
    
    def split_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """
        Split dataset into walk-forward train/validation splits.
        
        Args:
            df: Input dataset with timestamp and symbol columns
            
        Returns:
            List of split dictionaries with indices and metadata
        """
        start_time = time.time()
        
        # Validate input data
        df = self._validate_input_data(df)
        
        # Get date range
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        # Generate split windows
        split_windows = self._generate_split_windows(start_date, end_date)
        
        if not split_windows:
            raise ValueError("No valid split windows could be generated")
        
        # Process each split for each symbol
        all_splits = []
        symbols = df['symbol'].unique()
        
        logger.info("Processing splits",
                   split_windows=len(split_windows),
                   symbols=len(symbols))
        
        for split_info in split_windows:
            split_id = split_info['split_id']
            
            # Process each symbol separately
            symbol_splits = {}
            split_valid = True
            
            for symbol in symbols:
                # Extract split data
                train_data, val_data = self._extract_split_data(df, symbol, split_info)
                
                # Validate split quality
                if not self._validate_split_quality(train_data, val_data, split_info):
                    split_valid = False
                    break
                
                # Store indices
                symbol_splits[symbol] = {
                    'train_indices': train_data.index.tolist(),
                    'val_indices': val_data.index.tolist(),
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'train_date_range': {
                        'start': train_data['timestamp'].min().isoformat() if not train_data.empty else None,
                        'end': train_data['timestamp'].max().isoformat() if not train_data.empty else None
                    },
                    'val_date_range': {
                        'start': val_data['timestamp'].min().isoformat() if not val_data.empty else None,
                        'end': val_data['timestamp'].max().isoformat() if not val_data.empty else None
                    }
                }
                
                # Update statistics
                self.stats['total_train_samples'] += len(train_data)
                self.stats['total_val_samples'] += len(val_data)
            
            if split_valid:
                # Create complete split entry
                complete_split = {
                    'split_id': split_id,
                    'metadata': {
                        'train_start': split_info['train_start'].isoformat(),
                        'train_end': split_info['train_end'].isoformat(),
                        'val_start': split_info['val_start'].isoformat(),
                        'val_end': split_info['val_end'].isoformat(),
                        'gap_days': split_info['gap_days'],
                        'symbols': list(symbols)
                    },
                    'splits': symbol_splits
                }
                
                all_splits.append(complete_split)
                self.stats['valid_splits'] += 1
            
            self.stats['total_splits'] += 1
        
        # Analyze coverage and overlap
        coverage_stats = self._analyze_coverage_and_overlap(split_windows, df)
        self.stats.update(coverage_stats)
        
        # Update processing time
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info("Time series splitting completed",
                   total_splits=self.stats['total_splits'],
                   valid_splits=self.stats['valid_splits'],
                   coverage_ratio=f"{self.stats['coverage_ratio']:.1%}",
                   processing_time=f"{self.stats['processing_time']:.2f}s")
        
        return all_splits
    
    def print_statistics(self):
        """Print detailed splitting statistics."""
        print("\n" + "="*70)
        print("TIME SERIES SPLITTING STATISTICS")
        print("="*70)
        print(f"📊 Total splits generated: {self.stats['total_splits']}")
        print(f"✅ Valid splits:           {self.stats['valid_splits']}")
        print(f"🎯 Training samples:       {self.stats['total_train_samples']:,}")
        print(f"🎯 Validation samples:     {self.stats['total_val_samples']:,}")
        print(f"📈 Data coverage:          {self.stats['coverage_ratio']:.1%}")
        print(f"🔄 Data overlap:           {self.stats['overlap_ratio']:.1%}")
        print(f"⏱️  Processing time:       {self.stats['processing_time']:.2f}s")
        
        # Quality assessment
        print(f"\n📈 Quality Assessment:")
        if self.stats['valid_splits'] == 0:
            print(f"   ❌ No valid splits generated")
        elif self.stats['valid_splits'] < 3:
            print(f"   ⚠️  Few splits available ({self.stats['valid_splits']}) - consider longer data or smaller windows")
        else:
            print(f"   ✅ Good number of splits ({self.stats['valid_splits']})")
        
        if self.stats['coverage_ratio'] < 0.8:
            print(f"   ⚠️  Low data coverage ({self.stats['coverage_ratio']:.1%}) - consider overlapping splits")
        else:
            print(f"   ✅ Good data coverage ({self.stats['coverage_ratio']:.1%})")
        
        if self.stats['overlap_ratio'] > 0.5:
            print(f"   ⚠️  High overlap ({self.stats['overlap_ratio']:.1%}) - consider larger step size")
        elif self.stats['overlap_ratio'] > 0:
            print(f"   ✅ Controlled overlap ({self.stats['overlap_ratio']:.1%})")
        else:
            print(f"   ✅ No overlap (walk-forward)")
        
        print("="*70)


def save_splits(splits: List[Dict], output_dir: Union[str, Path], 
               dataset_name: str, output_format: str = 'indices'):
    """Save splits in various formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'indices':
        # Save as JSON with indices
        output_path = output_dir / f"{dataset_name}_splits.json"
        
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2, default=str)
        
        logger.info("Splits saved as indices",
                   path=str(output_path),
                   splits=len(splits))
    
    elif output_format == 'metadata':
        # Save only metadata without indices
        metadata_only = []
        for split in splits:
            metadata_split = {
                'split_id': split['split_id'],
                'metadata': split['metadata']
            }
            # Add summary statistics
            total_train = sum(s['train_samples'] for s in split['splits'].values())
            total_val = sum(s['val_samples'] for s in split['splits'].values())
            metadata_split['summary'] = {
                'total_train_samples': total_train,
                'total_val_samples': total_val,
                'symbols_count': len(split['splits'])
            }
            metadata_only.append(metadata_split)
        
        output_path = output_dir / f"{dataset_name}_split_metadata.json"
        
        with open(output_path, 'w') as f:
            json.dump(metadata_only, f, indent=2, default=str)
        
        logger.info("Split metadata saved",
                   path=str(output_path),
                   splits=len(metadata_only))
    
    elif output_format == 'data':
        # Save actual data files for each split (requires original dataset)
        logger.warning("Data format saving requires original dataset - not implemented in this function")
        raise NotImplementedError("Data format saving requires dataset parameter")
    
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def load_dataset(input_path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset for splitting."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    
    if input_path.suffix.lower() != '.parquet':
        raise ValueError(f"Dataset must be Parquet format: {input_path}")
    
    try:
        df = pd.read_parquet(input_path)
        logger.info("Dataset loaded",
                   file=str(input_path),
                   rows=len(df),
                   columns=len(df.columns),
                   memory_mb=f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def main():
    """Command-line interface for time series splitting."""
    parser = argparse.ArgumentParser(
        description="Create walk-forward time series splits for ML validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic walk-forward splitting
  python scripts/split_time_series.py \\
    --input data/datasets/eurusd_hour_dataset.parquet \\
    --train-years 2 --val-years 1

  # Advanced splitting with gaps
  python scripts/split_time_series.py \\
    --input data/datasets/eurusd_hour_dataset.parquet \\
    --train-years 3 --val-years 0.5 --step-years 1 --gap-days 7

  # Multiple datasets
  python scripts/split_time_series.py \\
    --input "data/datasets/*_dataset.parquet" \\
    --train-years 2 --val-years 1
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset file or glob pattern'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/splits',
        help='Output directory for splits (default: outputs/splits)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['indices', 'metadata', 'data'],
        default='indices',
        help='Output format: indices (JSON with indices), metadata (JSON without indices), data (separate files)'
    )
    
    # Splitting parameters
    parser.add_argument(
        '--train-years',
        type=float,
        default=2.0,
        help='Training window size in years (default: 2.0)'
    )
    
    parser.add_argument(
        '--val-years',
        type=float,
        default=1.0,
        help='Validation window size in years (default: 1.0)'
    )
    
    parser.add_argument(
        '--step-years',
        type=float,
        default=1.0,
        help='Step size between splits in years (default: 1.0)'
    )
    
    parser.add_argument(
        '--gap-days',
        type=int,
        default=0,
        help='Gap between train and validation in days (default: 0)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum samples per split (default: 100)'
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
        print("📊 ATRX Time Series Splitting System")
        print("="*50)
        
        # Initialize splitter
        splitter = TimeSeriesSplitter(
            train_years=args.train_years,
            val_years=args.val_years,
            step_years=args.step_years,
            gap_days=args.gap_days,
            min_samples_per_split=args.min_samples
        )
        
        # Handle glob patterns
        if '*' in args.input:
            input_files = glob(args.input)
            if not input_files:
                raise FileNotFoundError(f"No files found matching: {args.input}")
        else:
            input_files = [args.input]
        
        print(f"📁 Processing {len(input_files)} dataset(s)")
        
        successful = 0
        
        for input_file in sorted(input_files):
            print(f"\n📄 Processing: {input_file}")
            
            try:
                # Load dataset
                dataset = load_dataset(input_file)
                
                # Create splits
                splits = splitter.split_dataset(dataset)
                
                if not splits:
                    print(f"⚠️  No valid splits generated for {input_file}")
                    continue
                
                # Determine output path
                dataset_name = Path(input_file).stem.replace('_dataset', '')
                symbol_output_dir = Path(args.output_dir) / dataset_name
                
                # Check if output exists
                output_file = symbol_output_dir / f"{dataset_name}_splits.json"
                if output_file.exists() and not args.force:
                    print(f"⚠️  Output file exists: {output_file}")
                    print("   Use --force to overwrite")
                    continue
                
                # Save splits
                save_splits(splits, symbol_output_dir, dataset_name, args.output_format)
                
                # Print statistics
                splitter.print_statistics()
                
                print(f"✅ Splits saved to: {symbol_output_dir}")
                successful += 1
                
            except Exception as e:
                print(f"❌ Failed to process {input_file}: {e}")
                logger.exception("Processing failed")
                continue
        
        if successful > 0:
            print(f"\n🎉 Successfully processed {successful}/{len(input_files)} dataset(s)!")
        else:
            print(f"\n❌ No datasets were successfully processed")
            return 1
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Fatal error during splitting")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
