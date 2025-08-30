#!/usr/bin/env python3
"""
ATRX Dataset Assembly Tests
===========================

Comprehensive test suite for dataset assembly functionality.
Tests feature-label joining, NaN handling, and data validation.

Following ATRX debugging guidelines for thorough validation.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.assemble_dataset import DatasetAssembler, save_dataset, find_matching_files


class TestDatasetAssembler:
    """Test suite for DatasetAssembler functionality."""
    
    @pytest.fixture
    def basic_assembler(self):
        """Create a basic DatasetAssembler for testing."""
        return DatasetAssembler(min_samples=10, max_nan_ratio=0.2)
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample features data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='h', tz='UTC')
        
        # Create realistic feature data
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'symbol': ['EURUSD'] * 100,
            'close': np.random.normal(1.1, 0.01, 100),
            'RV_w': np.random.lognormal(-3, 0.5, 100),
            'ATR_w': np.random.lognormal(-4, 0.3, 100),
            'RSI_w': np.random.uniform(20, 80, 100),
            'feature_with_nans': [np.nan if i % 10 == 0 else np.random.normal(0, 1) for i in range(100)]
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_labels_data(self):
        """Create sample labels data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='h', tz='UTC')
        
        # Create labels that mostly match features (with some missing)
        np.random.seed(123)
        labels = np.random.choice([-1, 0, 1], size=95, p=[0.2, 0.6, 0.2])
        
        # Use slightly different timestamps to test join behavior
        label_dates = dates[:95]  # 5 fewer labels than features
        
        data = {
            'timestamp': label_dates,
            'symbol': ['EURUSD'] * 95,
            'label': labels
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def multi_symbol_features(self):
        """Create multi-symbol features data."""
        dates = pd.date_range('2020-01-01', periods=50, freq='h', tz='UTC')
        
        # Create data for two symbols
        eurusd_data = {
            'timestamp': dates,
            'symbol': ['EURUSD'] * 50,
            'close': np.random.normal(1.1, 0.01, 50),
            'RV_w': np.random.lognormal(-3, 0.5, 50)
        }
        
        gbpusd_data = {
            'timestamp': dates,
            'symbol': ['GBPUSD'] * 50,
            'close': np.random.normal(1.3, 0.01, 50),
            'RV_w': np.random.lognormal(-3, 0.5, 50)
        }
        
        eurusd_df = pd.DataFrame(eurusd_data)
        gbpusd_df = pd.DataFrame(gbpusd_data)
        
        return pd.concat([eurusd_df, gbpusd_df], ignore_index=True)
    
    @pytest.fixture
    def multi_symbol_labels(self):
        """Create multi-symbol labels data."""
        dates = pd.date_range('2020-01-01', periods=50, freq='h', tz='UTC')
        
        # Create labels for both symbols
        eurusd_labels = {
            'timestamp': dates,
            'symbol': ['EURUSD'] * 50,
            'label': np.random.choice([-1, 0, 1], 50)
        }
        
        gbpusd_labels = {
            'timestamp': dates,
            'symbol': ['GBPUSD'] * 50,
            'label': np.random.choice([-1, 0, 1], 50)
        }
        
        eurusd_df = pd.DataFrame(eurusd_labels)
        gbpusd_df = pd.DataFrame(gbpusd_labels)
        
        return pd.concat([eurusd_df, gbpusd_df], ignore_index=True)
    
    def test_assembler_initialization(self):
        """Test proper initialization of DatasetAssembler."""
        # Test default initialization
        assembler = DatasetAssembler()
        assert assembler.min_samples == 100
        assert assembler.max_nan_ratio == 0.1
        assert assembler.drop_features == []
        
        # Test custom initialization
        assembler = DatasetAssembler(
            min_samples=50,
            max_nan_ratio=0.05,
            drop_features=['feature1', 'feature2']
        )
        assert assembler.min_samples == 50
        assert assembler.max_nan_ratio == 0.05
        assert assembler.drop_features == ['feature1', 'feature2']
    
    def test_data_validation(self, basic_assembler):
        """Test input data validation."""
        # Test missing required columns in features
        incomplete_features = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'close': range(10)
            # Missing 'symbol' column
        })
        
        labels = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10,
            'label': [1] * 10
        })
        
        with pytest.raises(ValueError, match="Missing required feature columns"):
            basic_assembler._validate_data_structure(incomplete_features, labels)
        
        # Test missing required columns in labels
        features = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10,
            'close': range(10)
        })
        
        incomplete_labels = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10
            # Missing 'label' column
        })
        
        with pytest.raises(ValueError, match="Missing required label columns"):
            basic_assembler._validate_data_structure(features, incomplete_labels)
    
    def test_data_joining(self, basic_assembler, sample_features_data, sample_labels_data):
        """Test feature-label joining functionality."""
        # Test successful join
        joined_df = basic_assembler._join_data(sample_features_data, sample_labels_data)
        
        # Should have inner join result (95 rows, since labels has 95 rows)
        assert len(joined_df) == 95
        
        # Should have all feature columns plus label
        expected_columns = set(sample_features_data.columns).union(set(sample_labels_data.columns))
        assert set(joined_df.columns) == expected_columns
        
        # Check that join is properly aligned
        assert 'label' in joined_df.columns
        assert joined_df['label'].notna().all()
    
    def test_missing_data_analysis(self, basic_assembler, sample_features_data):
        """Test missing data analysis functionality."""
        # Add a column with high missing ratio
        test_data = sample_features_data.copy()
        test_data['high_missing'] = [np.nan] * 80 + [1] * 20  # 80% missing
        
        missing_stats = basic_assembler._analyze_missing_data(test_data)
        
        # Check that high missing column is detected
        assert 'high_missing' in missing_stats
        assert missing_stats['high_missing'] == 0.8
        
        # Check that normal columns have reasonable missing ratios
        assert missing_stats['feature_with_nans'] == 0.1  # 10% missing as designed
    
    def test_missing_data_handling(self, basic_assembler, sample_features_data, sample_labels_data):
        """Test missing data handling strategies."""
        # Join data first
        joined_df = basic_assembler._join_data(sample_features_data, sample_labels_data)
        
        # Handle missing data
        clean_df = basic_assembler._handle_missing_data(joined_df)
        
        # Should have removed rows with NaN values
        assert len(clean_df) < len(joined_df)
        
        # Should have no NaN values in final result
        assert not clean_df.isnull().any().any()
        
        # Check that feature with high missing ratio was handled
        if 'feature_with_nans' in clean_df.columns:
            # If kept, should have no NaN values
            assert not clean_df['feature_with_nans'].isnull().any()
    
    def test_high_missing_ratio_column_removal(self):
        """Test removal of columns with high missing ratios."""
        assembler = DatasetAssembler(max_nan_ratio=0.1)  # 10% threshold
        
        # Create data with a column that exceeds threshold
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='h'),
            'symbol': ['EURUSD'] * 100,
            'good_feature': range(100),
            'bad_feature': [np.nan] * 50 + list(range(50)),  # 50% missing
            'label': [1] * 100
        })
        
        clean_df = assembler._handle_missing_data(test_data)
        
        # Bad feature should be removed
        assert 'bad_feature' not in clean_df.columns
        assert 'good_feature' in clean_df.columns
    
    def test_multi_symbol_processing(self, basic_assembler, multi_symbol_features, multi_symbol_labels):
        """Test processing of multi-symbol datasets."""
        # Join multi-symbol data
        joined_df = basic_assembler._join_data(multi_symbol_features, multi_symbol_labels)
        
        # Should have data for both symbols
        symbols = joined_df['symbol'].unique()
        assert 'EURUSD' in symbols
        assert 'GBPUSD' in symbols
        
        # Should have expected number of rows (50 per symbol)
        assert len(joined_df) == 100
        
        # Handle missing data
        clean_df = basic_assembler._handle_missing_data(joined_df)
        
        # Should still have both symbols
        symbols = clean_df['symbol'].unique()
        assert 'EURUSD' in symbols
        assert 'GBPUSD' in symbols
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        assembler = DatasetAssembler(min_samples=100)
        
        # Create minimal dataset
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=50, freq='h'),
            'symbol': ['EURUSD'] * 50,
            'close': range(50),
            'label': [1] * 50
        })
        
        # Should raise error due to insufficient samples
        with pytest.raises(ValueError, match="Insufficient valid samples"):
            assembler._handle_missing_data(small_data)
    
    def test_full_assembly_workflow(self, basic_assembler):
        """Test the complete assembly workflow with temporary files."""
        # Create temporary feature and label files
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_features:
            features_path = f_features.name
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_labels:
            labels_path = f_labels.name
        
        try:
            # Create test data
            dates = pd.date_range('2020-01-01', periods=200, freq='h', tz='UTC')
            
            features_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': ['EURUSD'] * 200,
                'close': np.random.normal(1.1, 0.01, 200),
                'RV_w': np.random.lognormal(-3, 0.5, 200),
                'ATR_w': np.random.lognormal(-4, 0.3, 200)
            })
            
            labels_data = pd.DataFrame({
                'timestamp': dates[:180],  # Slightly fewer labels
                'symbol': ['EURUSD'] * 180,
                'label': np.random.choice([-1, 0, 1], 180)
            })
            
            # Save test data
            features_data.to_parquet(features_path, index=False)
            labels_data.to_parquet(labels_path, index=False)
            
            # Test assembly
            assembled_df = basic_assembler.assemble_dataset(features_path, labels_path)
            
            # Validate results
            assert len(assembled_df) > 0
            assert 'timestamp' in assembled_df.columns
            assert 'symbol' in assembled_df.columns
            assert 'label' in assembled_df.columns
            assert 'close' in assembled_df.columns
            
            # Check data quality
            assert not assembled_df.isnull().any().any()
            assert assembled_df['timestamp'].is_monotonic_increasing
            
            # Check statistics were collected
            assert basic_assembler.stats['features_loaded'] == 200
            assert basic_assembler.stats['labels_loaded'] == 180
            assert basic_assembler.stats['joined_samples'] == 180
            assert basic_assembler.stats['valid_samples'] > 0
            
        finally:
            # Clean up temporary files
            Path(features_path).unlink(missing_ok=True)
            Path(labels_path).unlink(missing_ok=True)
    
    def test_label_distribution_analysis(self, basic_assembler):
        """Test label distribution analysis and warnings."""
        # Create imbalanced label data
        imbalanced_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='h'),
            'symbol': ['EURUSD'] * 100,
            'close': range(100),
            'label': [1] * 95 + [-1] * 5  # Highly imbalanced
        })
        
        # This should complete without error and log warnings
        basic_assembler._validate_final_dataset(imbalanced_data)
    
    def test_temporal_ordering_validation(self, basic_assembler):
        """Test temporal ordering validation and correction."""
        # Create unordered data
        dates = pd.date_range('2020-01-01', periods=50, freq='h')
        unordered_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['EURUSD'] * 50,
            'close': range(50),
            'label': [1] * 50
        })
        
        # Shuffle the data to make it unordered
        unordered_data = unordered_data.sample(frac=1).reset_index(drop=True)
        
        # Validation should handle this gracefully
        basic_assembler._validate_final_dataset(unordered_data)


class TestDatasetAssemblyUtilities:
    """Test utility functions for dataset assembly."""
    
    def test_save_dataset_functionality(self):
        """Test dataset saving with metadata."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            output_path = f.name
        
        try:
            # Create test dataset
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
                'symbol': ['EURUSD'] * 10,
                'close': range(10),
                'label': [1] * 10
            })
            
            # Create test metadata
            metadata = {
                'created_at': '2023-01-01T00:00:00',
                'features_file': 'test_features.parquet',
                'labels_file': 'test_labels.parquet'
            }
            
            # Test saving
            save_dataset(test_data, output_path, metadata)
            
            # Verify file was created
            assert Path(output_path).exists()
            
            # Verify data can be loaded
            loaded_data = pd.read_parquet(output_path)
            assert len(loaded_data) == 10
            assert list(loaded_data.columns) == list(test_data.columns)
            
            # Check metadata file
            metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.json"
            assert metadata_path.exists()
            
            import json
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
            
            assert saved_metadata['features_file'] == 'test_features.parquet'
            
        finally:
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.json"
            metadata_path.unlink(missing_ok=True)
    
    def test_file_matching_functionality(self):
        """Test file pattern matching for batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "eurusd_features.parquet").touch()
            (temp_path / "gbpusd_features.parquet").touch()
            (temp_path / "eurusd_labels.parquet").touch()
            (temp_path / "gbpusd_labels.parquet").touch()
            (temp_path / "usdjpy_features.parquet").touch()  # No matching label
            
            # Test matching
            features_pattern = str(temp_path / "*_features.parquet")
            labels_pattern = str(temp_path / "*_labels.parquet")
            
            matches = find_matching_files(features_pattern, labels_pattern)
            
            # Should find 2 matching pairs
            assert len(matches) == 2
            
            # Check that matches are correct
            feature_files = [m[0] for m in matches]
            label_files = [m[1] for m in matches]
            
            assert any("eurusd" in f for f in feature_files)
            assert any("gbpusd" in f for f in feature_files)
            assert any("eurusd" in f for f in label_files)
            assert any("gbpusd" in f for f in label_files)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_file_not_found_errors(self):
        """Test handling of missing files."""
        assembler = DatasetAssembler()
        
        with pytest.raises(FileNotFoundError):
            assembler.assemble_dataset("nonexistent_features.parquet", "nonexistent_labels.parquet")
    
    def test_invalid_file_format_errors(self, basic_assembler):
        """Test handling of invalid file formats."""
        with tempfile.NamedTemporaryFile(suffix='.csv') as f:
            with pytest.raises(ValueError, match="must be Parquet format"):
                basic_assembler._validate_input_files(f.name, f.name)
    
    def test_no_common_symbols_error(self, basic_assembler):
        """Test error when no common symbols exist."""
        features = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10,
            'close': range(10)
        })
        
        labels = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['GBPUSD'] * 10,  # Different symbol
            'label': [1] * 10
        })
        
        with pytest.raises(ValueError, match="No common symbols"):
            basic_assembler._validate_data_structure(features, labels)
    
    def test_no_matching_timestamps_error(self, basic_assembler):
        """Test error when no timestamps match."""
        features = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10,
            'close': range(10)
        })
        
        labels = pd.DataFrame({
            'timestamp': pd.date_range('2021-01-01', periods=10, freq='h'),  # Different year
            'symbol': ['EURUSD'] * 10,
            'label': [1] * 10
        })
        
        with pytest.raises(ValueError, match="No matching timestamps"):
            basic_assembler._join_data(features, labels)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
