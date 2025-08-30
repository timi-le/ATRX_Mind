#!/usr/bin/env python3
"""
ATRX Dataset Assembly System
============================

Join features and labels to create ML-ready datasets following ATRX standards.
Handles data validation, NaN removal, and quality control for robust training.

Features:
- Inner join on timestamp/symbol for perfect alignment
- Comprehensive data validation and quality checks
- NaN handling with configurable strategies
- Memory-efficient processing for large datasets
- Detailed statistics and quality reporting

Following ATRX development standards for production-ready ML data preparation.

Usage:
    # Basic assembly
    python scripts/assemble_dataset.py --features data/features/eurusd_hour_features.parquet --labels data/labels/eurusd_hour_labels.parquet

    # Multiple datasets with custom output
    python scripts/assemble_dataset.py --features "data/features/*_features.parquet" --labels "data/labels/*_labels.parquet" --output-dir data/datasets/

    # Advanced options
    python scripts/assemble_dataset.py --features data/features/eurusd_hour_features.parquet --labels data/labels/eurusd_hour_labels.parquet --min-samples 1000 --max-nan-ratio 0.05
"""

import argparse
import logging
import time
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


class DatasetAssembler:
    """
    Advanced dataset assembly system for joining features and labels.
    
    Ensures data quality, handles missing values, and creates ML-ready datasets
    with comprehensive validation and reporting.
    """
    
    def __init__(self, min_samples: int = 100, max_nan_ratio: float = 0.1,
                 drop_features: Optional[List[str]] = None):
        """
        Initialize the dataset assembler.
        
        Args:
            min_samples: Minimum number of valid samples required
            max_nan_ratio: Maximum allowed ratio of NaN values per column
            drop_features: List of feature columns to exclude from final dataset
        """
        self.min_samples = min_samples
        self.max_nan_ratio = max_nan_ratio
        self.drop_features = drop_features or []
        
        # Statistics tracking
        self.stats = {
            'features_loaded': 0,
            'labels_loaded': 0,
            'joined_samples': 0,
            'valid_samples': 0,
            'features_dropped': 0,
            'nan_removed': 0,
            'processing_time': 0.0,
            'memory_mb': 0.0
        }
        
        logger.info("Dataset assembler initialized",
                   min_samples=min_samples,
                   max_nan_ratio=f"{max_nan_ratio:.1%}",
                   drop_features=len(self.drop_features))
    
    def _validate_input_files(self, features_path: Union[str, Path], 
                            labels_path: Union[str, Path]) -> Tuple[Path, Path]:
        """Validate input file paths."""
        features_path = Path(features_path)
        labels_path = Path(labels_path)
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        if features_path.suffix.lower() != '.parquet':
            raise ValueError(f"Features file must be Parquet format: {features_path}")
        
        if labels_path.suffix.lower() != '.parquet':
            raise ValueError(f"Labels file must be Parquet format: {labels_path}")
        
        return features_path, labels_path
    
    def _load_data(self, features_path: Path, labels_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load features and labels with validation."""
        logger.info("Loading data files",
                   features=str(features_path),
                   labels=str(labels_path))
        
        try:
            # Load features
            features_df = pd.read_parquet(features_path)
            self.stats['features_loaded'] = len(features_df)
            
            logger.info("Features loaded",
                       rows=len(features_df),
                       columns=len(features_df.columns),
                       memory_mb=f"{features_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
            
            # Load labels
            labels_df = pd.read_parquet(labels_path)
            self.stats['labels_loaded'] = len(labels_df)
            
            logger.info("Labels loaded",
                       rows=len(labels_df),
                       columns=len(labels_df.columns),
                       memory_mb=f"{labels_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data files: {e}")
        
        return features_df, labels_df
    
    def _validate_data_structure(self, features_df: pd.DataFrame, labels_df: pd.DataFrame):
        """Validate data structure and required columns."""
        # Check required columns for features
        required_feature_cols = ['timestamp', 'symbol']
        missing_feature_cols = [col for col in required_feature_cols if col not in features_df.columns]
        
        if missing_feature_cols:
            raise ValueError(f"Missing required feature columns: {missing_feature_cols}")
        
        # Check required columns for labels
        required_label_cols = ['timestamp', 'symbol', 'label']
        missing_label_cols = [col for col in required_label_cols if col not in labels_df.columns]
        
        if missing_label_cols:
            raise ValueError(f"Missing required label columns: {missing_label_cols}")
        
        # Validate timestamp columns
        if not pd.api.types.is_datetime64_any_dtype(features_df['timestamp']):
            logger.warning("Converting features timestamp to datetime")
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], utc=True)
        
        if not pd.api.types.is_datetime64_any_dtype(labels_df['timestamp']):
            logger.warning("Converting labels timestamp to datetime")
            labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'], utc=True)
        
        # Check symbol consistency
        feature_symbols = set(features_df['symbol'].unique())
        label_symbols = set(labels_df['symbol'].unique())
        
        if not feature_symbols.intersection(label_symbols):
            raise ValueError(f"No common symbols between features {feature_symbols} and labels {label_symbols}")
        
        common_symbols = feature_symbols.intersection(label_symbols)
        logger.info("Data validation completed",
                   common_symbols=list(common_symbols),
                   feature_symbols=len(feature_symbols),
                   label_symbols=len(label_symbols))
    
    def _join_data(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Join features and labels on timestamp and symbol."""
        logger.info("Joining features and labels")
        
        # Ensure data is sorted for efficient joining
        features_df = features_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        labels_df = labels_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Perform inner join to ensure perfect alignment
        joined_df = pd.merge(
            features_df,
            labels_df,
            on=['timestamp', 'symbol'],
            how='inner',
            suffixes=('', '_label')
        )
        
        self.stats['joined_samples'] = len(joined_df)
        
        logger.info("Data joined",
                   features_rows=len(features_df),
                   labels_rows=len(labels_df),
                   joined_rows=len(joined_df),
                   join_efficiency=f"{len(joined_df)/max(len(features_df), len(labels_df))*100:.1f}%")
        
        if len(joined_df) == 0:
            raise ValueError("No matching timestamps/symbols found between features and labels")
        
        return joined_df
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze missing data patterns."""
        missing_stats = {}
        
        # Critical columns that should not be dropped regardless of missing ratio
        critical_columns = ['timestamp', 'symbol', 'label']
        
        for col in df.columns:
            if col in critical_columns:
                continue
                
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df)
            missing_stats[col] = missing_ratio
        
        return missing_stats
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data according to configured strategy with retention optimization."""
        logger.info("Analyzing missing data patterns")
        
        # Analyze missing data
        missing_stats = self._analyze_missing_data(df)
        
        # Identify columns to drop due to high missing ratio
        cols_to_drop = []
        for col, ratio in missing_stats.items():
            if ratio > self.max_nan_ratio:
                cols_to_drop.append(col)
                logger.warning("Column will be dropped due to high missing ratio",
                              column=col, missing_ratio=f"{ratio:.1%}")
        
        # Add manually specified columns to drop
        cols_to_drop.extend(self.drop_features)
        cols_to_drop = list(set(cols_to_drop))  # Remove duplicates
        
        # Drop problematic columns
        if cols_to_drop:
            available_cols = [col for col in cols_to_drop if col in df.columns]
            if available_cols:
                df = df.drop(columns=available_cols)
                self.stats['features_dropped'] = len(available_cols)
                logger.info("Dropped features with high missing ratios",
                           dropped_features=available_cols)
        
        # Count rows before NaN handling
        initial_rows = len(df)
        
        # Only drop rows where the label is missing (critical for supervised learning)
        if 'label' in df.columns:
            label_na_mask = df['label'].isna()
            rows_with_label_na = label_na_mask.sum()
            
            if rows_with_label_na > 0:
                logger.info("Removing rows with missing labels",
                           rows_removed=rows_with_label_na)
                df = df[~label_na_mask]
        
        # For features, apply forward/backward fill instead of dropping rows
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'label']]
        if feature_cols:
            logger.info("Forward/backward filling remaining NaN values in features")
            for col in feature_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        final_rows = len(df)
        retention_rate = final_rows / initial_rows if initial_rows > 0 else 0.0
        
        self.stats['nan_removed'] = initial_rows - final_rows
        self.stats['valid_samples'] = final_rows
        self.stats['retention_rate'] = retention_rate
        
        # Calculate feature-specific retention (excluding VoV-filtered labels)
        features_retained = 1 - (self.stats.get('features_dropped', 0) / len(df.columns))
        
        # Check if low retention is due to VoV filtering (which is expected)
        if 'label' in df.columns and rows_with_label_na > 0:
            vov_filter_ratio = rows_with_label_na / initial_rows
            logger.info("Label filtering analysis",
                       vov_filtered_labels=rows_with_label_na,
                       vov_filter_ratio=f"{vov_filter_ratio:.1%}")
            
            # If most data loss is from VoV filtering (not feature issues), use relaxed threshold
            if vov_filter_ratio > 0.30:  # More than 30% filtered by VoV
                min_retention_rate = 0.50  # Relaxed to 50% when VoV filtering is active
                logger.info("VoV filtering detected: using relaxed retention threshold",
                           relaxed_threshold=f"{min_retention_rate:.1%}")
            else:
                min_retention_rate = 0.80  # Standard 80% for feature-based retention
        else:
            min_retention_rate = 0.80  # Standard 80% minimum retention
        
        if retention_rate < min_retention_rate:
            logger.error("Data retention rate too low",
                        retention_rate=f"{retention_rate:.1%}",
                        minimum_required=f"{min_retention_rate:.1%}")
            raise ValueError(f"Data retention rate {retention_rate:.1%} below minimum {min_retention_rate:.1%}")
        
        if final_rows < self.min_samples:
            raise ValueError(f"Insufficient valid samples: {final_rows} < {self.min_samples} required")
        
        logger.info("Missing data handled with retention optimization",
                   initial_rows=initial_rows,
                   final_rows=final_rows,
                   removed_rows=initial_rows - final_rows,
                   retention_rate=f"{retention_rate:.1%}")
        
        return df
    
    def _validate_final_dataset(self, df: pd.DataFrame):
        """Validate the final assembled dataset."""
        logger.info("Validating final dataset")
        
        # Check minimum samples requirement
        if len(df) < self.min_samples:
            raise ValueError(f"Final dataset has insufficient samples: {len(df)} < {self.min_samples}")
        
        # Check for any remaining NaN values
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            logger.error("Unexpected NaN values in final dataset", nan_counts=nan_counts.to_dict())
            raise ValueError("Final dataset contains NaN values")
        
        # Check label distribution
        if 'label' in df.columns:
            label_dist = df['label'].value_counts().sort_index()
            logger.info("Label distribution", distribution=label_dist.to_dict())
            
            # Warn if labels are heavily imbalanced
            if len(label_dist) > 1:
                max_count = label_dist.max()
                min_count = label_dist.min()
                imbalance_ratio = max_count / min_count
                
                if imbalance_ratio > 10:
                    logger.warning("Highly imbalanced labels detected",
                                  imbalance_ratio=f"{imbalance_ratio:.1f}:1")
        
        # Check temporal ordering
        if not df['timestamp'].is_monotonic_increasing:
            logger.warning("Dataset is not temporally ordered - sorting")
            df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Final memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        self.stats['memory_mb'] = memory_mb
        
        logger.info("Final dataset validation completed",
                   rows=len(df),
                   columns=len(df.columns),
                   memory_mb=f"{memory_mb:.2f}",
                   symbols=df['symbol'].nunique(),
                   date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}")
    
    def assemble_dataset(self, features_path: Union[str, Path], 
                        labels_path: Union[str, Path]) -> pd.DataFrame:
        """
        Assemble a complete dataset from features and labels.
        
        Args:
            features_path: Path to features Parquet file
            labels_path: Path to labels Parquet file
            
        Returns:
            Assembled and validated dataset
        """
        start_time = time.time()
        
        try:
            # Validate input paths
            features_path, labels_path = self._validate_input_files(features_path, labels_path)
            
            # Load data
            features_df, labels_df = self._load_data(features_path, labels_path)
            
            # Validate data structure
            self._validate_data_structure(features_df, labels_df)
            
            # Join data
            joined_df = self._join_data(features_df, labels_df)
            
            # Handle missing data
            clean_df = self._handle_missing_data(joined_df)
            
            # Final validation
            self._validate_final_dataset(clean_df)
            
            # Update processing time
            self.stats['processing_time'] = time.time() - start_time
            
            logger.info("Dataset assembly completed successfully",
                       processing_time=f"{self.stats['processing_time']:.2f}s")
            
            return clean_df
            
        except Exception as e:
            logger.error("Dataset assembly failed", error=str(e))
            raise
    
    def print_statistics(self):
        """Print detailed assembly statistics."""
        print("\n" + "="*70)
        print("DATASET ASSEMBLY STATISTICS")
        print("="*70)
        print(f"📊 Features loaded:        {self.stats['features_loaded']:,}")
        print(f"🏷️  Labels loaded:          {self.stats['labels_loaded']:,}")
        print(f"🔗 Joined samples:         {self.stats['joined_samples']:,}")
        print(f"✅ Valid samples:          {self.stats['valid_samples']:,}")
        print(f"🗑️  Features dropped:       {self.stats['features_dropped']}")
        print(f"🚫 NaN rows removed:       {self.stats['nan_removed']:,}")
        print(f"💾 Final memory usage:     {self.stats['memory_mb']:.2f} MB")
        print(f"⏱️  Processing time:       {self.stats['processing_time']:.2f}s")
        
        # Enhanced quality metrics with retention focus
        if self.stats['features_loaded'] > 0:
            join_efficiency = self.stats['joined_samples'] / self.stats['features_loaded'] * 100
            retention_rate = self.stats.get('retention_rate', 0.0)
            
            print(f"\n📈 Quality Metrics:")
            print(f"   Join efficiency:  {join_efficiency:.1f}%")
            print(f"   Data retention:   {retention_rate:.1%}")
            
            if join_efficiency < 50:
                print(f"   ⚠️  Low join efficiency - check timestamp alignment")
            
            # Enhanced retention assessment with VoV context
            nan_removed = self.stats.get('nan_removed', 0)
            joined_samples = self.stats.get('joined_samples', 0)
            vov_filter_detected = joined_samples > 0 and (nan_removed / joined_samples) > 0.30
            
            if vov_filter_detected:
                if retention_rate >= 0.50:
                    print(f"   ✅ Good retention with VoV filtering (≥50%)")
                else:
                    print(f"   ⚠️  Low retention even with VoV filtering (<50%)")
                print(f"   📊 VoV filter removed: {(nan_removed / joined_samples):.1%} of data")
            else:
                if retention_rate >= 0.80:
                    print(f"   ✅ Excellent retention (≥80%)")
                elif retention_rate >= 0.70:
                    print(f"   ⚠️  Moderate retention (70-80%)")
                else:
                    print(f"   ❌ Poor retention (<70%) - review NaN handling strategy")
            
            # Show NaN loss breakdown
            nan_removed = self.stats.get('nan_removed', 0)
            joined_samples = self.stats.get('joined_samples', 0)
            if joined_samples > 0:
                nan_loss_rate = nan_removed / joined_samples
                print(f"   NaN loss rate:    {nan_loss_rate:.1%}")
        
        print("="*70)


def save_dataset(df: pd.DataFrame, output_path: Union[str, Path], 
                metadata: Optional[Dict] = None):
    """Save assembled dataset with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save main dataset
        df.to_parquet(output_path, compression='snappy', index=False)
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("Metadata saved", path=str(metadata_path))
        
        logger.info("Dataset saved",
                   path=str(output_path),
                   rows=len(df),
                   size_mb=f"{output_path.stat().st_size / 1024**2:.2f}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save dataset: {e}")


def find_matching_files(features_pattern: str, labels_pattern: str) -> List[Tuple[str, str]]:
    """Find matching feature and label files based on patterns."""
    feature_files = glob(features_pattern)
    label_files = glob(labels_pattern)
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found matching: {features_pattern}")
    
    if not label_files:
        raise FileNotFoundError(f"No label files found matching: {labels_pattern}")
    
    # Match files based on common basename patterns
    matches = []
    
    for feature_file in feature_files:
        feature_path = Path(feature_file)
        # Extract base name (remove _features suffix)
        base_name = feature_path.stem.replace('_features', '')
        
        # Look for corresponding label file
        for label_file in label_files:
            label_path = Path(label_file)
            if base_name in label_path.stem:
                matches.append((feature_file, label_file))
                break
        else:
            logger.warning("No matching label file found", feature_file=feature_file)
    
    return matches


def main():
    """Command-line interface for dataset assembly."""
    parser = argparse.ArgumentParser(
        description="Assemble ML-ready datasets by joining features and labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic assembly
  python scripts/assemble_dataset.py \\
    --features data/features/eurusd_hour_features.parquet \\
    --labels data/labels/eurusd_hour_labels.parquet

  # Multiple datasets with glob patterns
  python scripts/assemble_dataset.py \\
    --features "data/features/*_features.parquet" \\
    --labels "data/labels/*_labels.parquet" \\
    --output-dir data/datasets/

  # Custom quality settings
  python scripts/assemble_dataset.py \\
    --features data/features/eurusd_hour_features.parquet \\
    --labels data/labels/eurusd_hour_labels.parquet \\
    --min-samples 5000 \\
    --max-nan-ratio 0.02 \\
    --drop-features quote_slope bid_ask_imbalance
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Features file path or glob pattern'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Labels file path or glob pattern'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output dataset file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/datasets',
        help='Output directory for datasets (default: data/datasets)'
    )
    
    # Quality control arguments
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum number of valid samples required (default: 100)'
    )
    
    parser.add_argument(
        '--max-nan-ratio',
        type=float,
        default=0.1,
        help='Maximum allowed NaN ratio per column (default: 0.1)'
    )
    
    parser.add_argument(
        '--drop-features',
        nargs='*',
        default=[],
        help='Feature columns to drop from final dataset'
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
        print("🔗 ATRX Dataset Assembly System")
        print("="*50)
        
        # Initialize assembler
        assembler = DatasetAssembler(
            min_samples=args.min_samples,
            max_nan_ratio=args.max_nan_ratio,
            drop_features=args.drop_features
        )
        
        # Handle glob patterns
        if '*' in args.features or '*' in args.labels:
            matches = find_matching_files(args.features, args.labels)
            if not matches:
                raise ValueError("No matching feature/label file pairs found")
            
            print(f"📁 Found {len(matches)} matching file pair(s)")
        else:
            matches = [(args.features, args.labels)]
        
        successful = 0
        
        for features_file, labels_file in matches:
            print(f"\n📄 Processing pair:")
            print(f"   Features: {features_file}")
            print(f"   Labels:   {labels_file}")
            
            try:
                # Assemble dataset
                dataset_df = assembler.assemble_dataset(features_file, labels_file)
                
                # Determine output path
                if args.output:
                    output_path = args.output
                else:
                    # Auto-generate output path
                    base_name = Path(features_file).stem.replace('_features', '')
                    output_path = Path(args.output_dir) / f"{base_name}_dataset.parquet"
                
                # Check if output exists
                if Path(output_path).exists() and not args.force:
                    print(f"⚠️  Output file exists: {output_path}")
                    print("   Use --force to overwrite")
                    continue
                
                # Create metadata
                metadata = {
                    'features_file': str(features_file),
                    'labels_file': str(labels_file),
                    'assembly_stats': assembler.stats,
                    'created_at': pd.Timestamp.now().isoformat(),
                    'columns': list(dataset_df.columns),
                    'symbols': list(dataset_df['symbol'].unique()),
                    'date_range': {
                        'start': dataset_df['timestamp'].min().isoformat(),
                        'end': dataset_df['timestamp'].max().isoformat()
                    }
                }
                
                # Save dataset
                save_dataset(dataset_df, output_path, metadata)
                
                # Print statistics
                assembler.print_statistics()
                
                print(f"✅ Dataset saved to: {output_path}")
                successful += 1
                
            except Exception as e:
                print(f"❌ Failed to process pair: {e}")
                logger.exception("Processing failed")
                continue
        
        if successful > 0:
            print(f"\n🎉 Successfully processed {successful}/{len(matches)} dataset(s)!")
        else:
            print(f"\n❌ No datasets were successfully processed")
            return 1
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Fatal error during assembly")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
