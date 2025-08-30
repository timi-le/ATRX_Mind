#!/usr/bin/env python3
"""
Normalize FX Data to Canonical Parquet Format
==============================================

Standardizes all FX CSV files into canonical Parquet format under data/parquet/.
Supports two schema families:
- Schema A: OHLC format (EURUSD_H1.csv, USDJPY_H1.csv, GBPUSD_H1.csv)
- Schema B: Bid/Ask format (eurusd_hour.csv)

Features:
- Timestamp parsing with UTC enforcement
- Symbol inference from filename
- Deduplication and sorting
- Comprehensive validation
- Progress tracking and logging

Usage:
    poetry run python scripts/normalize_to_parquet.py
    poetry run python scripts/normalize_to_parquet.py --input-dir data/fx_data --output-dir data/parquet
    poetry run python scripts/normalize_to_parquet.py --file data/fx_data/eurusd_hour_cleaned.csv --force
"""

import argparse
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)


class FXDataNormalizer:
    """
    Comprehensive FX data normalization to canonical Parquet format.
    
    Handles both OHLC and Bid/Ask schema families with robust validation,
    timestamp processing, and symbol inference.
    """
    
    # Schema A: OHLC format columns
    OHLC_SCHEMA = [
        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 
        'tick_volume', 'real_volume', 'spread'
    ]
    
    # Schema B: Bid/Ask format columns
    BIDASK_SCHEMA = [
        'timestamp', 'symbol', 
        'bid_open', 'bid_high', 'bid_low', 'bid_close',
        'ask_open', 'ask_high', 'ask_low', 'ask_close',
        'mid_open', 'mid_high', 'mid_low', 'mid_close',
        'spread_close', 'bid_change', 'ask_change'
    ]
    
    # Symbol mapping patterns
    SYMBOL_PATTERNS = {
        r'EURUSD': 'EURUSD',
        r'GBPUSD': 'GBPUSD', 
        r'USDJPY': 'USDJPY',
        r'eurusd': 'EURUSD'  # Normalize case
    }
    
    def __init__(self, output_dir: str = "data/parquet"):
        """Initialize normalizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'files_processed': 0,
            'files_succeeded': 0,
            'files_failed': 0,
            'total_rows_processed': 0,
            'total_rows_output': 0
        }
        
        logger.info("FX Data Normalizer initialized", output_dir=str(self.output_dir))
    
    def infer_symbol_from_filename(self, filename: str) -> str:
        """
        Infer trading symbol from filename using pattern matching.
        
        Args:
            filename: Input filename (e.g., 'EURUSD_H1.csv', 'eurusd_hour.csv')
            
        Returns:
            Standardized symbol (e.g., 'EURUSD')
            
        Raises:
            ValueError: If symbol cannot be inferred
        """
        filename_upper = filename.upper()
        
        for pattern, symbol in self.SYMBOL_PATTERNS.items():
            if re.search(pattern, filename_upper):
                logger.debug("Symbol inferred", filename=filename, symbol=symbol, pattern=pattern)
                return symbol
                
        raise ValueError(f"Cannot infer symbol from filename: {filename}")
    
    def detect_schema_type(self, df: pd.DataFrame) -> str:
        """
        Detect whether CSV follows OHLC (Schema A) or Bid/Ask (Schema B) format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            'ohlc' or 'bidask'
            
        Raises:
            ValueError: If schema cannot be determined
        """
        columns = set(df.columns)
        
        # Check for Bid/Ask schema markers
        bidask_markers = {'BO', 'BH', 'BL', 'BC', 'AO', 'AH', 'AL', 'AC'}
        if bidask_markers.issubset(columns):
            logger.debug("Detected Bid/Ask schema", columns=list(columns))
            return 'bidask'
        
        # Check for OHLC schema markers
        ohlc_markers = {'open', 'high', 'low', 'close'}
        if ohlc_markers.issubset(columns):
            logger.debug("Detected OHLC schema", columns=list(columns))
            return 'ohlc'
            
        raise ValueError(f"Cannot determine schema type from columns: {list(columns)}")
    
    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and standardize timestamps to UTC timezone.
        
        Handles multiple timestamp formats:
        - Unix timestamps in 'time' column
        - Separate 'Date' and 'Time' columns
        - Combined datetime strings
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized 'timestamp' column
        """
        df = df.copy()
        
        if 'timestamp' in df.columns:
            # Already has timestamp column
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
        elif 'time' in df.columns:
            # Unix timestamp format (common in MT5 data)
            if df['time'].dtype in ['int64', 'float64']:
                df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
            else:
                df['timestamp'] = pd.to_datetime(df['time'], utc=True)
                
        elif 'Date' in df.columns and 'Time' in df.columns:
            # Separate date and time columns (bid/ask format)
            df['timestamp'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str), 
                utc=True
            )
            
        elif 'date' in df.columns:
            # Single date column
            df['timestamp'] = pd.to_datetime(df['date'], utc=True)
            
        else:
            raise ValueError("Cannot find timestamp columns in data")
        
        # Validate timestamp range (should be reasonable for FX data)
        min_date = pd.Timestamp('1970-01-01', tz='UTC')
        max_date = pd.Timestamp('2030-01-01', tz='UTC')
        
        invalid_timestamps = (df['timestamp'] < min_date) | (df['timestamp'] > max_date)
        if invalid_timestamps.any():
            logger.warning(
                "Invalid timestamps found", 
                count=invalid_timestamps.sum(),
                min_ts=df['timestamp'].min(),
                max_ts=df['timestamp'].max()
            )
            # Remove invalid timestamps
            df = df[~invalid_timestamps]
        
        logger.debug(
            "Timestamps parsed", 
            count=len(df),
            date_range=(df['timestamp'].min(), df['timestamp'].max())
        )
        
        return df
    
    def normalize_ohlc_schema(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize OHLC format (Schema A) to canonical structure.
        
        Input columns: time, open, high, low, close, tick_volume, spread, real_volume, date
        Output columns: timestamp, symbol, open, high, low, close, tick_volume, real_volume, spread
        
        Args:
            df: Input DataFrame with OHLC data
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Normalized DataFrame with canonical OHLC schema
        """
        logger.info("Normalizing OHLC schema", symbol=symbol, input_rows=len(df))
        
        # Parse timestamps
        df = self.parse_timestamps(df)
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Ensure required numeric columns exist and have proper dtypes
        required_numeric = ['open', 'high', 'low', 'close']
        for col in required_numeric:
            if col not in df.columns:
                raise ValueError(f"Required OHLC column missing: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle optional columns with defaults
        if 'tick_volume' not in df.columns:
            df['tick_volume'] = 0.0
        else:
            df['tick_volume'] = pd.to_numeric(df['tick_volume'], errors='coerce').fillna(0.0)
            
        if 'real_volume' not in df.columns:
            df['real_volume'] = 0.0
        else:
            df['real_volume'] = pd.to_numeric(df['real_volume'], errors='coerce').fillna(0.0)
            
        if 'spread' not in df.columns:
            df['spread'] = 0.0
        else:
            df['spread'] = pd.to_numeric(df['spread'], errors='coerce').fillna(0.0)
        
        # Validate OHLC relationships
        invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                       (df['low'] > df['open']) | (df['low'] > df['close'])
        
        if invalid_ohlc.any():
            logger.warning("Invalid OHLC relationships found", count=invalid_ohlc.sum())
            df = df[~invalid_ohlc]
        
        # Select and reorder columns
        normalized_df = df[self.OHLC_SCHEMA].copy()
        
        logger.info("OHLC normalization completed", output_rows=len(normalized_df))
        return normalized_df
    
    def normalize_bidask_schema(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize Bid/Ask format (Schema B) to canonical structure.
        
        Input columns: Date, Time, BO, BH, BL, BC, BCh, AO, AH, AL, AC, ACh
        Output columns: timestamp, symbol, bid_*, ask_*, mid_*, spread_close, bid_change, ask_change
        
        Args:
            df: Input DataFrame with Bid/Ask data
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Normalized DataFrame with canonical Bid/Ask schema
        """
        logger.info("Normalizing Bid/Ask schema", symbol=symbol, input_rows=len(df))
        
        # Parse timestamps
        df = self.parse_timestamps(df)
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Map bid/ask columns to canonical names
        bid_mapping = {
            'BO': 'bid_open',
            'BH': 'bid_high', 
            'BL': 'bid_low',
            'BC': 'bid_close'
        }
        
        ask_mapping = {
            'AO': 'ask_open',
            'AH': 'ask_high',
            'AL': 'ask_low', 
            'AC': 'ask_close'
        }
        
        # Validate required columns exist
        required_cols = list(bid_mapping.keys()) + list(ask_mapping.keys())
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Required Bid/Ask columns missing: {missing_cols}")
        
        # Apply mappings and convert to numeric
        for old_col, new_col in bid_mapping.items():
            df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
            
        for old_col, new_col in ask_mapping.items():
            df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        # Calculate mid prices (mean of bid and ask)
        df['mid_open'] = (df['bid_open'] + df['ask_open']) / 2
        df['mid_high'] = (df['bid_high'] + df['ask_high']) / 2
        df['mid_low'] = (df['bid_low'] + df['ask_low']) / 2
        df['mid_close'] = (df['bid_close'] + df['ask_close']) / 2
        
        # Calculate spread at close
        df['spread_close'] = df['ask_close'] - df['bid_close']
        
        # Handle change columns (BCh, ACh) if present
        if 'BCh' in df.columns:
            df['bid_change'] = pd.to_numeric(df['BCh'], errors='coerce').fillna(0.0)
        else:
            df['bid_change'] = 0.0
            
        if 'ACh' in df.columns:
            df['ask_change'] = pd.to_numeric(df['ACh'], errors='coerce').fillna(0.0)
        else:
            df['ask_change'] = 0.0
        
        # Validate bid/ask relationships
        invalid_spread = df['spread_close'] < 0
        if invalid_spread.any():
            logger.warning("Negative spreads found", count=invalid_spread.sum())
            df = df[~invalid_spread]
        
        # Select and reorder columns
        normalized_df = df[self.BIDASK_SCHEMA].copy()
        
        logger.info("Bid/Ask normalization completed", output_rows=len(normalized_df))
        return normalized_df
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final cleaning and validation steps.
        
        - Sort by timestamp
        - Remove duplicates 
        - Drop rows with null timestamps
        - Validate data ranges
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Cleaned and validated DataFrame
        """
        initial_rows = len(df)
        
        # Remove rows with null timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicate timestamps (keep first occurrence)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        
        if rows_removed > 0:
            logger.info("Data cleaning completed", 
                       initial_rows=initial_rows, 
                       final_rows=final_rows, 
                       rows_removed=rows_removed)
        
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, output_path: Path, 
                       compression: str = 'snappy') -> bool:
        """
        Save DataFrame to Parquet format with optimal settings.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            compression: Compression algorithm ('snappy', 'gzip', 'brotli')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to PyArrow table for better type handling
            table = pa.Table.from_pandas(df)
            
            # Write with optimal settings for time series data
            pq.write_table(
                table, 
                output_path,
                compression=compression,
                row_group_size=50000,  # Optimize for time series queries
                use_dictionary=True,   # Compress symbol strings
                write_statistics=True  # Enable query optimization
            )
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info("Parquet file saved", 
                       path=str(output_path), 
                       rows=len(df),
                       size_mb=f"{file_size:.2f}")
            
            return True
            
        except Exception as e:
            logger.error("Failed to save Parquet file", 
                        path=str(output_path), 
                        error=str(e))
            return False
    
    def process_file(self, input_path: Union[str, Path], force: bool = False) -> bool:
        """
        Process a single CSV file to normalized Parquet format.
        
        Args:
            input_path: Path to input CSV file
            force: Whether to overwrite existing output files
            
        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error("Input file not found", path=str(input_path))
            return False
        
        # Generate output path
        base_name = input_path.stem.replace('_cleaned', '').replace('_original_backup', '')
        output_path = self.output_dir / f"{base_name}.parquet"
        
        if output_path.exists() and not force:
            logger.info("Output file exists, skipping", 
                       input=str(input_path), 
                       output=str(output_path))
            return True
        
        try:
            self.stats['files_processed'] += 1
            
            logger.info("Processing file", input=str(input_path))
            
            # Load CSV data
            df = pd.read_csv(input_path)
            initial_rows = len(df)
            self.stats['total_rows_processed'] += initial_rows
            
            logger.info("CSV loaded", rows=initial_rows, columns=list(df.columns))
            
            # Infer symbol from filename
            symbol = self.infer_symbol_from_filename(input_path.name)
            
            # Detect schema type
            schema_type = self.detect_schema_type(df)
            
            # Normalize based on schema
            if schema_type == 'ohlc':
                normalized_df = self.normalize_ohlc_schema(df, symbol)
            elif schema_type == 'bidask':
                normalized_df = self.normalize_bidask_schema(df, symbol)
            else:
                raise ValueError(f"Unsupported schema type: {schema_type}")
            
            # Clean and validate
            final_df = self.clean_and_validate(normalized_df)
            final_rows = len(final_df)
            self.stats['total_rows_output'] += final_rows
            
            # Save to Parquet
            success = self.save_to_parquet(final_df, output_path)
            
            if success:
                self.stats['files_succeeded'] += 1
                logger.info("File processing completed", 
                           input=str(input_path),
                           output=str(output_path),
                           schema=schema_type,
                           symbol=symbol,
                           input_rows=initial_rows,
                           output_rows=final_rows)
                return True
            else:
                self.stats['files_failed'] += 1
                return False
                
        except Exception as e:
            self.stats['files_failed'] += 1
            logger.error("File processing failed", 
                        input=str(input_path), 
                        error=str(e),
                        exc_info=True)
            return False
    
    def process_directory(self, input_dir: Union[str, Path], 
                         pattern: str = "*.csv", force: bool = False) -> int:
        """
        Process all CSV files in a directory.
        
        Args:
            input_dir: Input directory path
            pattern: File pattern to match (default: "*.csv")
            force: Whether to overwrite existing output files
            
        Returns:
            Number of files successfully processed
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            logger.error("Input directory not found", path=str(input_dir))
            return 0
        
        # Find CSV files, excluding backups and already processed files
        csv_files = []
        for file_path in input_dir.glob(pattern):
            if any(skip in file_path.name for skip in ['_backup', '_original']):
                logger.debug("Skipping backup file", file=file_path.name)
                continue
            csv_files.append(file_path)
        
        if not csv_files:
            logger.warning("No CSV files found", directory=str(input_dir), pattern=pattern)
            return 0
        
        logger.info("Processing directory", 
                   directory=str(input_dir), 
                   files_found=len(csv_files))
        
        successful = 0
        for csv_file in sorted(csv_files):
            if self.process_file(csv_file, force=force):
                successful += 1
        
        return successful
    
    def print_summary(self):
        """Print processing summary statistics."""
        print("\n" + "="*60)
        print("FX DATA NORMALIZATION SUMMARY")
        print("="*60)
        print(f"📁 Files processed:    {self.stats['files_processed']}")
        print(f"✅ Files succeeded:    {self.stats['files_succeeded']}")
        print(f"❌ Files failed:       {self.stats['files_failed']}")
        print(f"📊 Total rows input:   {self.stats['total_rows_processed']:,}")
        print(f"📊 Total rows output:  {self.stats['total_rows_output']:,}")
        
        if self.stats['total_rows_processed'] > 0:
            retention_rate = (self.stats['total_rows_output'] / self.stats['total_rows_processed']) * 100
            print(f"📈 Data retention:     {retention_rate:.1f}%")
        
        print(f"📁 Output directory:   {self.output_dir}")
        print("="*60)
        
        if self.stats['files_succeeded'] > 0:
            print(f"\n🎉 SUCCESS! {self.stats['files_succeeded']} file(s) normalized to Parquet format")
            print("🚀 Data is ready for high-performance ML training!")
        elif self.stats['files_failed'] > 0:
            print(f"\n⚠️ {self.stats['files_failed']} file(s) failed processing")


def main():
    """Command-line interface for FX data normalization."""
    parser = argparse.ArgumentParser(
        description="Normalize FX CSV data to canonical Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in fx_data directory
  python scripts/normalize_to_parquet.py

  # Process specific file
  python scripts/normalize_to_parquet.py --file data/fx_data/eurusd_hour_cleaned.csv

  # Force overwrite existing files
  python scripts/normalize_to_parquet.py --force

  # Custom input/output directories  
  python scripts/normalize_to_parquet.py --input-dir data/fx_data --output-dir data/parquet
        """
    )
    
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='data/fx_data',
        help='Input directory containing CSV files (default: data/fx_data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str, 
        default='data/parquet',
        help='Output directory for Parquet files (default: data/parquet)'
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
        '--pattern',
        type=str,
        default='*.csv',
        help='File pattern to match in directory mode (default: *.csv)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize normalizer
    normalizer = FXDataNormalizer(output_dir=args.output_dir)
    
    print("🔄 FX Data Normalization to Canonical Parquet Format")
    print("="*60)
    
    try:
        if args.file:
            # Process single file
            print(f"📄 Processing file: {args.file}")
            success = normalizer.process_file(args.file, force=args.force)
            if not success:
                print(f"❌ Failed to process {args.file}")
                return 1
        else:
            # Process directory
            print(f"📁 Processing directory: {args.input_dir}")
            print(f"🎯 Output directory: {args.output_dir}")
            print(f"🔍 Pattern: {args.pattern}")
            
            successful = normalizer.process_directory(
                args.input_dir, 
                pattern=args.pattern, 
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
        normalizer.print_summary()
    
    return 0


if __name__ == "__main__":
    exit(main())
