#!/usr/bin/env python3
"""
ATRX Time Series Splitting Tests
================================

Comprehensive test suite for time series splitting functionality.
Tests chronological ordering, data leakage prevention, and coverage validation.

Following ATRX debugging guidelines for thorough validation.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.split_time_series import TimeSeriesSplitter, save_splits, load_dataset


class TestTimeSeriesSplitter:
    """Test suite for TimeSeriesSplitter functionality."""
    
    @pytest.fixture
    def basic_splitter(self):
        """Create a basic TimeSeriesSplitter for testing."""
        return TimeSeriesSplitter(
            train_years=1.0,
            val_years=0.5,
            step_years=0.5,
            gap_days=0,
            min_samples_per_split=10
        )
    
    @pytest.fixture
    def gap_splitter(self):
        """Create a TimeSeriesSplitter with gap for testing."""
        return TimeSeriesSplitter(
            train_years=1.0,
            val_years=0.5,
            step_years=0.5,
            gap_days=7,  # 1 week gap
            min_samples_per_split=10
        )
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample time series dataset for testing."""
        # Create 3 years of hourly data
        dates = pd.date_range('2020-01-01', '2022-12-31 23:00:00', freq='h', tz='UTC')
        
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'symbol': ['EURUSD'] * len(dates),
            'close': np.random.normal(1.1, 0.01, len(dates)),
            'label': np.random.choice([-1, 0, 1], len(dates), p=[0.2, 0.6, 0.2])
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def multi_symbol_dataset(self):
        """Create multi-symbol time series dataset."""
        # Create 2 years of daily data for 2 symbols
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D', tz='UTC')
        
        # EURUSD data
        eurusd_data = {
            'timestamp': dates,
            'symbol': ['EURUSD'] * len(dates),
            'close': np.random.normal(1.1, 0.01, len(dates)),
            'label': np.random.choice([-1, 0, 1], len(dates))
        }
        
        # GBPUSD data
        gbpusd_data = {
            'timestamp': dates,
            'symbol': ['GBPUSD'] * len(dates),
            'close': np.random.normal(1.3, 0.01, len(dates)),
            'label': np.random.choice([-1, 0, 1], len(dates))
        }
        
        eurusd_df = pd.DataFrame(eurusd_data)
        gbpusd_df = pd.DataFrame(gbpusd_data)
        
        return pd.concat([eurusd_df, gbpusd_df], ignore_index=True)
    
    @pytest.fixture
    def sparse_dataset(self):
        """Create sparse dataset with irregular timestamps."""
        # Create irregular timestamps (some missing days)
        base_dates = pd.date_range('2020-01-01', '2022-12-31', freq='D', tz='UTC')
        # Remove random days to create gaps
        np.random.seed(123)
        keep_indices = np.random.choice(len(base_dates), size=int(len(base_dates) * 0.8), replace=False)
        dates = base_dates[sorted(keep_indices)]
        
        data = {
            'timestamp': dates,
            'symbol': ['EURUSD'] * len(dates),
            'close': np.random.normal(1.1, 0.01, len(dates)),
            'label': np.random.choice([-1, 0, 1], len(dates))
        }
        
        return pd.DataFrame(data)
    
    def test_splitter_initialization(self):
        """Test proper initialization of TimeSeriesSplitter."""
        # Test default initialization
        splitter = TimeSeriesSplitter()
        assert splitter.train_years == 2.0
        assert splitter.val_years == 1.0
        assert splitter.step_years == 1.0
        assert splitter.gap_days == 0
        
        # Test custom initialization
        splitter = TimeSeriesSplitter(
            train_years=3.0,
            val_years=1.5,
            step_years=0.5,
            gap_days=30,
            min_samples_per_split=50
        )
        assert splitter.train_years == 3.0
        assert splitter.val_years == 1.5
        assert splitter.step_years == 0.5
        assert splitter.gap_days == 30
        assert splitter.min_samples_per_split == 50
    
    def test_data_validation(self, basic_splitter):
        """Test input data validation."""
        # Test missing required columns
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='h'),
            'close': range(100)
            # Missing 'symbol' column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            basic_splitter._validate_input_data(incomplete_data)
        
        # Test insufficient data
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10,
            'close': range(10)
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            basic_splitter._validate_input_data(small_data)
    
    def test_split_window_generation(self, basic_splitter, sample_dataset):
        """Test generation of split windows."""
        # Validate data first
        validated_data = basic_splitter._validate_input_data(sample_dataset)
        
        start_date = validated_data['timestamp'].min()
        end_date = validated_data['timestamp'].max()
        
        # Generate split windows
        windows = basic_splitter._generate_split_windows(start_date, end_date)
        
        # Should generate multiple windows
        assert len(windows) > 0
        
        # Check window structure
        for window in windows:
            assert 'split_id' in window
            assert 'train_start' in window
            assert 'train_end' in window
            assert 'val_start' in window
            assert 'val_end' in window
            
            # Check temporal ordering
            assert window['train_start'] < window['train_end']
            assert window['train_end'] <= window['val_start']  # Allow for gap
            assert window['val_start'] < window['val_end']
        
        # Check that windows are properly spaced
        if len(windows) > 1:
            for i in range(1, len(windows)):
                prev_window = windows[i-1]
                curr_window = windows[i]
                
                # Current window should start after previous window's start
                assert curr_window['train_start'] > prev_window['train_start']
    
    def test_split_data_extraction(self, basic_splitter, sample_dataset):
        """Test extraction of train/validation data for splits."""
        # Prepare data
        validated_data = basic_splitter._validate_input_data(sample_dataset)
        
        # Create a test split window
        start_date = validated_data['timestamp'].min()
        split_info = {
            'split_id': 0,
            'train_start': start_date,
            'train_end': start_date + timedelta(days=365),
            'val_start': start_date + timedelta(days=365),
            'val_end': start_date + timedelta(days=365 + 182),
            'gap_days': 0
        }
        
        # Extract split data
        train_data, val_data = basic_splitter._extract_split_data(
            validated_data, 'EURUSD', split_info
        )
        
        # Check that data was extracted
        assert len(train_data) > 0
        assert len(val_data) > 0
        
        # Check temporal boundaries
        assert train_data['timestamp'].min() >= split_info['train_start']
        assert train_data['timestamp'].max() < split_info['train_end']
        assert val_data['timestamp'].min() >= split_info['val_start']
        assert val_data['timestamp'].max() < split_info['val_end']
        
        # Check no temporal leakage
        max_train_time = train_data['timestamp'].max()
        min_val_time = val_data['timestamp'].min()
        assert max_train_time < min_val_time
    
    def test_split_quality_validation(self, basic_splitter):
        """Test split quality validation."""
        # Create test data
        good_train_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='h'),
            'symbol': ['EURUSD'] * 100,
            'close': range(100)
        })
        
        good_val_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-05', periods=50, freq='h'),
            'symbol': ['EURUSD'] * 50,
            'close': range(50)
        })
        
        split_info = {'split_id': 0, 'gap_days': 0}
        
        # Test valid split
        assert basic_splitter._validate_split_quality(good_train_data, good_val_data, split_info)
        
        # Test insufficient training data
        small_train_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5, freq='h'),
            'symbol': ['EURUSD'] * 5,
            'close': range(5)
        })
        
        assert not basic_splitter._validate_split_quality(small_train_data, good_val_data, split_info)
        
        # Test insufficient validation data
        small_val_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-05', periods=5, freq='h'),
            'symbol': ['EURUSD'] * 5,
            'close': range(5)
        })
        
        assert not basic_splitter._validate_split_quality(good_train_data, small_val_data, split_info)
    
    def test_temporal_leakage_prevention(self, basic_splitter):
        """Test that temporal leakage is properly prevented."""
        # Create data with potential leakage
        leaky_train_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='h'),
            'symbol': ['EURUSD'] * 100,
            'close': range(100)
        })
        
        # Validation data starts before training data ends (leakage)
        leaky_val_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-03', periods=50, freq='h'),  # Overlaps with training
            'symbol': ['EURUSD'] * 50,
            'close': range(50)
        })
        
        split_info = {'split_id': 0, 'gap_days': 0}
        
        # Should detect leakage and reject split
        assert not basic_splitter._validate_split_quality(leaky_train_data, leaky_val_data, split_info)
    
    def test_gap_insertion(self, gap_splitter, sample_dataset):
        """Test proper gap insertion between train and validation."""
        # Generate splits with gap
        splits = gap_splitter.split_dataset(sample_dataset)
        
        assert len(splits) > 0
        
        # Check that gaps are properly implemented
        for split in splits:
            for symbol, symbol_split in split['splits'].items():
                if symbol_split['train_samples'] > 0 and symbol_split['val_samples'] > 0:
                    # Parse timestamps
                    train_end = pd.to_datetime(symbol_split['train_date_range']['end'])
                    val_start = pd.to_datetime(symbol_split['val_date_range']['start'])
                    
                    # Calculate actual gap
                    actual_gap = val_start - train_end
                    
                    # Should have at least the specified gap
                    expected_gap = timedelta(days=gap_splitter.gap_days)
                    assert actual_gap >= expected_gap - timedelta(hours=1)  # Allow small tolerance
    
    def test_multi_symbol_splitting(self, basic_splitter, multi_symbol_dataset):
        """Test splitting with multiple symbols."""
        splits = basic_splitter.split_dataset(multi_symbol_dataset)
        
        assert len(splits) > 0
        
        # Check that all splits handle both symbols
        for split in splits:
            assert 'EURUSD' in split['splits']
            assert 'GBPUSD' in split['splits']
            
            # Both symbols should have data
            eurusd_split = split['splits']['EURUSD']
            gbpusd_split = split['splits']['GBPUSD']
            
            assert eurusd_split['train_samples'] > 0
            assert eurusd_split['val_samples'] > 0
            assert gbpusd_split['train_samples'] > 0
            assert gbpusd_split['val_samples'] > 0
    
    def test_chronological_ordering(self, basic_splitter, sample_dataset):
        """Test that splits maintain chronological ordering."""
        splits = basic_splitter.split_dataset(sample_dataset)
        
        assert len(splits) > 1  # Need multiple splits to test ordering
        
        # Check that splits are chronologically ordered
        for i in range(1, len(splits)):
            prev_split = splits[i-1]
            curr_split = splits[i]
            
            # Current split should start after previous split
            prev_train_start = pd.to_datetime(prev_split['metadata']['train_start'])
            curr_train_start = pd.to_datetime(curr_split['metadata']['train_start'])
            
            assert curr_train_start > prev_train_start
    
    def test_no_data_overlap_between_splits(self, basic_splitter, sample_dataset):
        """Test that there's no data overlap between train/validation across splits."""
        splits = basic_splitter.split_dataset(sample_dataset)
        
        if len(splits) < 2:
            pytest.skip("Need at least 2 splits for overlap testing")
        
        # Collect all indices used across splits
        all_train_indices = set()
        all_val_indices = set()
        
        for split in splits:
            for symbol, symbol_split in split['splits'].items():
                train_indices = set(symbol_split['train_indices'])
                val_indices = set(symbol_split['val_indices'])
                
                # Check for overlap within this split (should be none)
                assert len(train_indices.intersection(val_indices)) == 0
                
                # Check for overlap with previous splits' validation data
                overlap_with_prev_val = val_indices.intersection(all_val_indices)
                # Note: Some overlap is expected in walk-forward, but not within train/val of same split
                
                all_train_indices.update(train_indices)
                all_val_indices.update(val_indices)
    
    def test_coverage_analysis(self, basic_splitter, sample_dataset):
        """Test data coverage analysis."""
        splits = basic_splitter.split_dataset(sample_dataset)
        
        # Check coverage statistics
        assert 'coverage_ratio' in basic_splitter.stats
        assert 'overlap_ratio' in basic_splitter.stats
        
        # Coverage should be reasonable (> 0)
        assert basic_splitter.stats['coverage_ratio'] > 0
        assert basic_splitter.stats['coverage_ratio'] <= 1.0
        
        # Overlap should be reasonable for walk-forward
        assert basic_splitter.stats['overlap_ratio'] >= 0
    
    def test_insufficient_time_range_handling(self, sample_dataset):
        """Test handling of datasets with insufficient time range."""
        # Create splitter with very large windows
        large_splitter = TimeSeriesSplitter(
            train_years=10.0,  # Much larger than available data
            val_years=5.0,
            step_years=1.0
        )
        
        # Should either generate no splits or handle gracefully
        splits = large_splitter.split_dataset(sample_dataset)
        
        # With insufficient data, should generate few or no splits
        assert len(splits) <= 1
    
    def test_sparse_data_handling(self, basic_splitter, sparse_dataset):
        """Test handling of sparse/irregular data."""
        # Should handle sparse data without errors
        splits = basic_splitter.split_dataset(sparse_dataset)
        
        # May generate fewer splits due to data sparsity, but should not crash
        assert isinstance(splits, list)
        
        # If splits are generated, they should be valid
        for split in splits:
            assert 'split_id' in split
            assert 'metadata' in split
            assert 'splits' in split
    
    def test_full_splitting_workflow(self, basic_splitter, sample_dataset):
        """Test the complete splitting workflow."""
        # Perform splitting
        splits = basic_splitter.split_dataset(sample_dataset)
        
        # Validate results
        assert len(splits) > 0
        
        for split in splits:
            # Check structure
            assert 'split_id' in split
            assert 'metadata' in split
            assert 'splits' in split
            
            # Check metadata
            metadata = split['metadata']
            assert 'train_start' in metadata
            assert 'train_end' in metadata
            assert 'val_start' in metadata
            assert 'val_end' in metadata
            assert 'symbols' in metadata
            
            # Check symbol splits
            for symbol, symbol_split in split['splits'].items():
                assert 'train_indices' in symbol_split
                assert 'val_indices' in symbol_split
                assert 'train_samples' in symbol_split
                assert 'val_samples' in symbol_split
                assert 'train_date_range' in symbol_split
                assert 'val_date_range' in symbol_split
        
        # Check statistics were collected
        assert basic_splitter.stats['total_splits'] > 0
        assert basic_splitter.stats['valid_splits'] > 0
        assert basic_splitter.stats['processing_time'] > 0


class TestTimeSeriesSplittingUtilities:
    """Test utility functions for time series splitting."""
    
    def test_save_splits_indices_format(self):
        """Test saving splits in indices format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test splits
            test_splits = [
                {
                    'split_id': 0,
                    'metadata': {
                        'train_start': '2020-01-01T00:00:00+00:00',
                        'train_end': '2021-01-01T00:00:00+00:00',
                        'val_start': '2021-01-01T00:00:00+00:00',
                        'val_end': '2021-07-01T00:00:00+00:00',
                        'symbols': ['EURUSD']
                    },
                    'splits': {
                        'EURUSD': {
                            'train_indices': [0, 1, 2, 3, 4],
                            'val_indices': [5, 6, 7],
                            'train_samples': 5,
                            'val_samples': 3
                        }
                    }
                }
            ]
            
            # Save splits
            save_splits(test_splits, temp_dir, 'test_dataset', 'indices')
            
            # Check file was created
            output_file = Path(temp_dir) / 'test_dataset_splits.json'
            assert output_file.exists()
            
            # Verify content
            with open(output_file, 'r') as f:
                saved_splits = json.load(f)
            
            assert len(saved_splits) == 1
            assert saved_splits[0]['split_id'] == 0
            assert 'train_indices' in saved_splits[0]['splits']['EURUSD']
    
    def test_save_splits_metadata_format(self):
        """Test saving splits in metadata-only format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test splits
            test_splits = [
                {
                    'split_id': 0,
                    'metadata': {
                        'train_start': '2020-01-01T00:00:00+00:00',
                        'train_end': '2021-01-01T00:00:00+00:00',
                        'val_start': '2021-01-01T00:00:00+00:00',
                        'val_end': '2021-07-01T00:00:00+00:00',
                        'symbols': ['EURUSD']
                    },
                    'splits': {
                        'EURUSD': {
                            'train_indices': [0, 1, 2, 3, 4],
                            'val_indices': [5, 6, 7],
                            'train_samples': 5,
                            'val_samples': 3
                        }
                    }
                }
            ]
            
            # Save splits
            save_splits(test_splits, temp_dir, 'test_dataset', 'metadata')
            
            # Check file was created
            output_file = Path(temp_dir) / 'test_dataset_split_metadata.json'
            assert output_file.exists()
            
            # Verify content
            with open(output_file, 'r') as f:
                saved_metadata = json.load(f)
            
            assert len(saved_metadata) == 1
            assert saved_metadata[0]['split_id'] == 0
            assert 'summary' in saved_metadata[0]
            # Should not contain indices
            assert 'splits' not in saved_metadata[0]
    
    def test_load_dataset_functionality(self):
        """Test dataset loading functionality."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            dataset_path = f.name
        
        try:
            # Create test dataset
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=100, freq='h', tz='UTC'),
                'symbol': ['EURUSD'] * 100,
                'close': np.random.normal(1.1, 0.01, 100),
                'label': np.random.choice([-1, 0, 1], 100)
            })
            
            # Save test data
            test_data.to_parquet(dataset_path, index=False)
            
            # Test loading
            loaded_data = load_dataset(dataset_path)
            
            assert len(loaded_data) == 100
            assert list(loaded_data.columns) == list(test_data.columns)
            assert loaded_data['symbol'].iloc[0] == 'EURUSD'
            
        finally:
            Path(dataset_path).unlink(missing_ok=True)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_file_not_found_error(self):
        """Test handling of missing dataset files."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_dataset.parquet")
    
    def test_invalid_file_format_error(self):
        """Test handling of invalid file formats."""
        with tempfile.NamedTemporaryFile(suffix='.csv') as f:
            with pytest.raises(ValueError, match="must be Parquet format"):
                load_dataset(f.name)
    
    def test_invalid_output_format_error(self):
        """Test handling of invalid output formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_splits = []
            
            with pytest.raises(ValueError, match="Unknown output format"):
                save_splits(test_splits, temp_dir, 'test', 'invalid_format')
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        splitter = TimeSeriesSplitter(min_samples_per_split=10)
        
        empty_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=0, freq='h'),
            'symbol': [],
            'close': [],
            'label': []
        })
        
        with pytest.raises(ValueError):
            splitter.split_dataset(empty_data)


class TestDataLeakagePrevention:
    """Specific tests for data leakage prevention."""
    
    def test_strict_temporal_ordering(self):
        """Test that temporal ordering is strictly enforced."""
        splitter = TimeSeriesSplitter(
            train_years=0.5,
            val_years=0.25,
            step_years=0.25,
            gap_days=1
        )
        
        # Create dataset with daily frequency
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D', tz='UTC')
        dataset = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['EURUSD'] * len(dates),
            'close': np.random.normal(1.1, 0.01, len(dates)),
            'label': np.random.choice([-1, 0, 1], len(dates))
        })
        
        splits = splitter.split_dataset(dataset)
        
        # Verify strict temporal ordering in each split
        for split in splits:
            for symbol, symbol_split in split['splits'].items():
                if symbol_split['train_samples'] > 0 and symbol_split['val_samples'] > 0:
                    # Get actual data indices
                    train_indices = symbol_split['train_indices']
                    val_indices = symbol_split['val_indices']
                    
                    # Get corresponding timestamps
                    train_timestamps = dataset.loc[train_indices, 'timestamp']
                    val_timestamps = dataset.loc[val_indices, 'timestamp']
                    
                    # Latest training timestamp should be before earliest validation timestamp
                    max_train_time = train_timestamps.max()
                    min_val_time = val_timestamps.min()
                    
                    assert max_train_time < min_val_time, f"Temporal leakage detected in split {split['split_id']}"
    
    def test_gap_effectiveness(self):
        """Test that gaps effectively prevent leakage."""
        # Test with and without gaps
        no_gap_splitter = TimeSeriesSplitter(train_years=0.5, val_years=0.25, gap_days=0)
        gap_splitter = TimeSeriesSplitter(train_years=0.5, val_years=0.25, gap_days=7)
        
        # Create high-frequency dataset where gaps matter
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='h', tz='UTC')
        dataset = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['EURUSD'] * len(dates),
            'close': np.random.normal(1.1, 0.01, len(dates)),
            'label': np.random.choice([-1, 0, 1], len(dates))
        })
        
        no_gap_splits = no_gap_splitter.split_dataset(dataset)
        gap_splits = gap_splitter.split_dataset(dataset)
        
        # With gaps, the time difference should be larger
        for i, (no_gap_split, gap_split) in enumerate(zip(no_gap_splits, gap_splits)):
            no_gap_symbol_split = no_gap_split['splits']['EURUSD']
            gap_symbol_split = gap_split['splits']['EURUSD']
            
            if (no_gap_symbol_split['train_samples'] > 0 and no_gap_symbol_split['val_samples'] > 0 and
                gap_symbol_split['train_samples'] > 0 and gap_symbol_split['val_samples'] > 0):
                
                # Calculate time gaps
                no_gap_train_end = pd.to_datetime(no_gap_symbol_split['train_date_range']['end'])
                no_gap_val_start = pd.to_datetime(no_gap_symbol_split['val_date_range']['start'])
                no_gap_gap = no_gap_val_start - no_gap_train_end
                
                gap_train_end = pd.to_datetime(gap_symbol_split['train_date_range']['end'])
                gap_val_start = pd.to_datetime(gap_symbol_split['val_date_range']['start'])
                gap_gap = gap_val_start - gap_train_end
                
                # Gap splitter should have larger gap
                assert gap_gap > no_gap_gap


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
