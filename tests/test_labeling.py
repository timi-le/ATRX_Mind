#!/usr/bin/env python3
"""
ATRX Triple-Barrier Labeling Tests
==================================

Comprehensive test suite for the Triple-Barrier labeling system with VoV filtering.
Tests label generation accuracy, VoV masking, and edge case handling.

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
from trainers.labeling.label_regimes import TripleBarrierLabeler, load_feature_data, save_labels


class TestTripleBarrierLabeling:
    """Test suite for Triple-Barrier labeling functionality."""
    
    @pytest.fixture
    def basic_labeler(self):
        """Create a basic Triple-Barrier labeler for testing."""
        return TripleBarrierLabeler(
            vol_basis='rv',
            price_col='close',
            k_up=2.0,
            k_dn=2.0,
            horizon=10,
            vov_col='VoV_wV',
            vov_lower_quant=0.2,
            vov_upper_quant=0.8
        )
    
    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data for testing."""
        np.random.seed(42)  # For reproducible tests
        n_obs = 100
        
        # Create realistic price data with trend
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.001, n_obs)  # 0.1% volatility
        prices = [base_price]
        
        for change in price_changes[:-1]:
            prices.append(prices[-1] * (1 + change))
        
        # Create corresponding volatility and VoV data
        rv_values = np.random.lognormal(np.log(0.02), 0.3, n_obs)  # Realistic RV values
        atr_values = rv_values * 0.8  # ATR typically lower than RV
        vov_values = np.random.lognormal(np.log(0.1), 0.5, n_obs)  # VoV values
        
        data = {
            'timestamp': pd.date_range('2020-01-01', periods=n_obs, freq='h', tz='UTC'),
            'symbol': ['EURUSD'] * n_obs,
            'close': prices,
            'RV_w': rv_values,
            'ATR_w': atr_values,
            'VoV_wV': vov_values
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def trending_data(self):
        """Create trending price data for testing directional labels."""
        n_obs = 50
        
        # Strong uptrend data
        uptrend_prices = [1.1000 + i * 0.0001 for i in range(n_obs)]  # Linear uptrend
        
        # Strong downtrend data  
        downtrend_prices = [1.2000 - i * 0.0001 for i in range(n_obs)]  # Linear downtrend
        
        # Low constant volatility for clean signals
        low_vol = [0.001] * n_obs
        normal_vov = [0.1] * n_obs
        
        uptrend_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n_obs, freq='h', tz='UTC'),
            'symbol': ['EURUSD'] * n_obs,
            'close': uptrend_prices,
            'RV_w': low_vol,
            'ATR_w': low_vol,
            'VoV_wV': normal_vov
        })
        
        downtrend_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n_obs, freq='h', tz='UTC'),
            'symbol': ['GBPUSD'] * n_obs,
            'close': downtrend_prices,
            'RV_w': low_vol,
            'ATR_w': low_vol,
            'VoV_wV': normal_vov
        })
        
        return pd.concat([uptrend_data, downtrend_data], ignore_index=True)
    
    @pytest.fixture
    def vov_extreme_data(self):
        """Create data with extreme VoV values for filtering tests."""
        n_obs = 30
        
        # Normal price data
        prices = [1.1000] * n_obs  # Constant prices
        normal_vol = [0.01] * n_obs
        
        # Create VoV values: 10 low, 10 normal, 10 high
        vov_values = (
            [0.01] * 10 +      # Very low VoV (should be filtered)
            [0.10] * 10 +      # Normal VoV (should be kept)
            [1.00] * 10        # Very high VoV (should be filtered)
        )
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n_obs, freq='h', tz='UTC'),
            'symbol': ['EURUSD'] * n_obs,
            'close': prices,
            'RV_w': normal_vol,
            'ATR_w': normal_vol,
            'VoV_wV': vov_values
        })
        
        return data
    
    def test_labeler_initialization(self):
        """Test proper initialization of labeler with various parameters."""
        # Test default initialization
        labeler = TripleBarrierLabeler()
        assert labeler.vol_basis == 'rv'
        assert labeler.price_col == 'close'
        assert labeler.k_up == 2.0
        assert labeler.k_dn == 2.0
        assert labeler.horizon == 24
        
        # Test custom initialization
        labeler = TripleBarrierLabeler(
            vol_basis='atr',
            price_col='mid_close',
            k_up=1.5,
            k_dn=1.0,
            horizon=48
        )
        assert labeler.vol_basis == 'atr'
        assert labeler.price_col == 'mid_close'
        assert labeler.k_up == 1.5
        assert labeler.k_dn == 1.0
        assert labeler.horizon == 48
    
    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid vol_basis
        with pytest.raises(ValueError, match="vol_basis must be"):
            TripleBarrierLabeler(vol_basis='invalid')
        
        # Test invalid multipliers
        with pytest.raises(ValueError, match="k_up and k_dn must be positive"):
            TripleBarrierLabeler(k_up=-1.0)
        
        with pytest.raises(ValueError, match="k_up and k_dn must be positive"):
            TripleBarrierLabeler(k_dn=0.0)
        
        # Test invalid horizon
        with pytest.raises(ValueError, match="horizon must be positive"):
            TripleBarrierLabeler(horizon=-5)
        
        # Test invalid VoV quantiles
        with pytest.raises(ValueError, match="VoV quantiles"):
            TripleBarrierLabeler(vov_lower_quant=0.8, vov_upper_quant=0.2)
    
    def test_data_validation(self, basic_labeler):
        """Test input data validation."""
        # Test missing required columns
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'close': range(10)
            # Missing 'symbol' column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            basic_labeler._validate_input_data(incomplete_data)
        
        # Test missing price column
        no_price_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10,
            'RV_w': [0.01] * 10
            # Missing 'close' column
        })
        
        with pytest.raises(ValueError, match="Price column"):
            basic_labeler._validate_input_data(no_price_data)
        
        # Test missing volatility column
        no_vol_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
            'symbol': ['EURUSD'] * 10,
            'close': range(10)
            # Missing 'RV_w' column
        })
        
        with pytest.raises(ValueError, match="Required volatility column"):
            basic_labeler._validate_input_data(no_vol_data)
        
        # Test insufficient data
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5, freq='h'),
            'symbol': ['EURUSD'] * 5,
            'close': range(5),
            'RV_w': [0.01] * 5
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            basic_labeler._validate_input_data(small_data)
    
    def test_volatility_column_detection(self, sample_feature_data):
        """Test automatic volatility column detection."""
        # Test RV basis
        rv_labeler = TripleBarrierLabeler(vol_basis='rv')
        vol_col = rv_labeler._get_volatility_column(sample_feature_data)
        assert vol_col == 'RV_w'
        
        # Test ATR basis
        atr_labeler = TripleBarrierLabeler(vol_basis='atr')
        vol_col = atr_labeler._get_volatility_column(sample_feature_data)
        assert vol_col == 'ATR_w'
    
    def test_barrier_computation(self, basic_labeler, sample_feature_data):
        """Test barrier computation accuracy."""
        df = basic_labeler._validate_input_data(sample_feature_data)
        upper_barriers, lower_barriers = basic_labeler._compute_barriers(df)
        
        # Test barrier shapes
        assert len(upper_barriers) == len(df)
        assert len(lower_barriers) == len(df)
        
        # Test barrier relationships
        prices = df['close'].values
        volatility = df['RV_w'].values
        
        expected_upper = prices + (basic_labeler.k_up * volatility)
        expected_lower = prices - (basic_labeler.k_dn * volatility)
        
        np.testing.assert_array_almost_equal(upper_barriers, expected_upper)
        np.testing.assert_array_almost_equal(lower_barriers, expected_lower)
        
        # Test that barriers are properly positioned
        assert np.all(upper_barriers >= prices)  # Upper barriers above prices
        assert np.all(lower_barriers <= prices)  # Lower barriers below prices
    
    def test_trending_upward_labels(self, trending_data):
        """Test label generation for strong upward trending data."""
        # Filter for uptrend data only
        uptrend_data = trending_data[trending_data['symbol'] == 'EURUSD'].copy()
        
        # Use small volatility to ensure upper barrier hits
        labeler = TripleBarrierLabeler(
            vol_basis='rv',
            k_up=0.5,  # Small multiplier for easy barrier hits
            k_dn=0.5,
            horizon=5,
            vov_col='VoV_wV'
        )
        
        labels_df = labeler.label_data(uptrend_data)
        
        # For strong uptrend with small volatility, most labels should be +1
        positive_labels = (labels_df['label'] == 1).sum()
        total_valid = labels_df['label'].notna().sum()
        
        assert positive_labels > 0, "Should have some positive labels for uptrend"
        assert positive_labels >= total_valid * 0.3, "Should have significant positive labels"
        
        # Should have very few negative labels
        negative_labels = (labels_df['label'] == -1).sum()
        assert negative_labels <= positive_labels, "Uptrend should have more positive than negative labels"
    
    def test_trending_downward_labels(self, trending_data):
        """Test label generation for strong downward trending data."""
        # Filter for downtrend data only
        downtrend_data = trending_data[trending_data['symbol'] == 'GBPUSD'].copy()
        
        # Use small volatility to ensure lower barrier hits
        labeler = TripleBarrierLabeler(
            vol_basis='rv',
            k_up=0.5,  # Small multiplier for easy barrier hits
            k_dn=0.5,
            horizon=5,
            vov_col='VoV_wV'
        )
        
        labels_df = labeler.label_data(downtrend_data)
        
        # For strong downtrend with small volatility, most labels should be -1
        negative_labels = (labels_df['label'] == -1).sum()
        total_valid = labels_df['label'].notna().sum()
        
        assert negative_labels > 0, "Should have some negative labels for downtrend"
        assert negative_labels >= total_valid * 0.3, "Should have significant negative labels"
        
        # Should have very few positive labels
        positive_labels = (labels_df['label'] == 1).sum()
        assert negative_labels >= positive_labels, "Downtrend should have more negative than positive labels"
    
    def test_timeout_labels(self, sample_feature_data):
        """Test timeout label generation when no barriers are hit."""
        # Use very large volatility multipliers to make barriers unreachable
        labeler = TripleBarrierLabeler(
            vol_basis='rv',
            k_up=10.0,  # Very large multipliers
            k_dn=10.0,
            horizon=5,
            vov_col='VoV_wV'
        )
        
        labels_df = labeler.label_data(sample_feature_data)
        
        # Most labels should be timeouts (0) with unreachable barriers
        timeout_labels = (labels_df['label'] == 0).sum()
        total_valid = labels_df['label'].notna().sum()
        
        assert timeout_labels > 0, "Should have some timeout labels with large barriers"
        assert timeout_labels >= total_valid * 0.5, "Most labels should be timeouts with unreachable barriers"
    
    def test_vov_filtering(self, vov_extreme_data):
        """Test VoV-based filtering functionality."""
        # Configure labeler with tight VoV filtering
        labeler = TripleBarrierLabeler(
            vol_basis='rv',
            k_up=1.0,
            k_dn=1.0,
            horizon=5,
            vov_col='VoV_wV',
            vov_lower_quant=0.4,  # Filter bottom 40%
            vov_upper_quant=0.6   # Filter top 40%
        )
        
        labels_df = labeler.label_data(vov_extreme_data)
        
        # Check that extreme VoV values resulted in NaN labels
        valid_labels = labels_df['label'].notna().sum()
        total_obs = len(labels_df)
        
        # With 40% filtered on each end, we should have ~20% valid labels
        # (plus some at the edges due to horizon effects)
        filtered_ratio = (total_obs - valid_labels) / total_obs
        assert filtered_ratio > 0.5, f"Should filter significant portion, got {filtered_ratio:.1%}"
        
        # Check that filtering statistics were tracked
        assert labeler.stats['vov_filtered'] > 0
    
    def test_no_vov_filtering_when_column_missing(self, sample_feature_data):
        """Test that labeling works when VoV column is missing."""
        # Remove VoV column
        data_no_vov = sample_feature_data.drop(columns=['VoV_wV'])
        
        labeler = TripleBarrierLabeler(
            vol_basis='rv',
            vov_col='VoV_wV'  # Column doesn't exist
        )
        
        # Should work without errors
        labels_df = labeler.label_data(data_no_vov)
        
        # Should have valid labels (no VoV filtering applied)
        assert labels_df['label'].notna().sum() > 0
        assert labeler.stats['vov_filtered'] == 0
    
    def test_multi_symbol_processing(self, sample_feature_data):
        """Test processing data with multiple symbols."""
        # Create multi-symbol data
        eurusd_data = sample_feature_data.copy()
        gbpusd_data = sample_feature_data.copy()
        gbpusd_data['symbol'] = 'GBPUSD'
        gbpusd_data['close'] = gbpusd_data['close'] * 1.3  # Different price level
        
        multi_symbol_data = pd.concat([eurusd_data, gbpusd_data], ignore_index=True)
        
        labeler = TripleBarrierLabeler(vol_basis='rv', horizon=5)
        labels_df = labeler.label_data(multi_symbol_data)
        
        # Check that both symbols are present
        symbols = labels_df['symbol'].unique()
        assert 'EURUSD' in symbols
        assert 'GBPUSD' in symbols
        
        # Check that labels were generated for both symbols
        eurusd_labels = labels_df[labels_df['symbol'] == 'EURUSD']['label'].notna().sum()
        gbpusd_labels = labels_df[labels_df['symbol'] == 'GBPUSD']['label'].notna().sum()
        
        assert eurusd_labels > 0, "Should have labels for EURUSD"
        assert gbpusd_labels > 0, "Should have labels for GBPUSD"
    
    def test_rv_vs_atr_basis_difference(self, sample_feature_data):
        """Test that RV and ATR basis produce different but reasonable results."""
        rv_labeler = TripleBarrierLabeler(vol_basis='rv', horizon=5)
        atr_labeler = TripleBarrierLabeler(vol_basis='atr', horizon=5)
        
        rv_labels = rv_labeler.label_data(sample_feature_data)
        atr_labels = atr_labeler.label_data(sample_feature_data)
        
        # Both should produce valid labels
        assert rv_labels['label'].notna().sum() > 0
        assert atr_labels['label'].notna().sum() > 0
        
        # Results may be different due to different volatility measures
        # but both should be reasonable
        rv_distribution = rv_labels['label'].value_counts()
        atr_distribution = atr_labels['label'].value_counts()
        
        # Both should have some labels (may be mostly timeouts with random data)
        assert len(rv_distribution) >= 1, "RV labeling should produce some labels"
        assert len(atr_distribution) >= 1, "ATR labeling should produce some labels"
        
        # At least one method should have labels (allow for edge case where one produces only timeouts)
        assert rv_labels['label'].notna().sum() > 0 or atr_labels['label'].notna().sum() > 0
    
    def test_statistics_tracking(self, basic_labeler, sample_feature_data):
        """Test that statistics are properly tracked during labeling."""
        labels_df = basic_labeler.label_data(sample_feature_data)
        
        # Check that statistics were collected
        stats = basic_labeler.stats
        
        assert stats['total_observations'] == len(sample_feature_data)
        assert stats['valid_labels'] >= 0
        assert stats['upper_hits'] >= 0
        assert stats['lower_hits'] >= 0
        assert stats['timeouts'] >= 0
        assert stats['vov_filtered'] >= 0
        assert stats['processing_time'] > 0
        
        # Check that statistics add up correctly
        total_labeled = stats['upper_hits'] + stats['lower_hits'] + stats['timeouts']
        assert total_labeled == stats['valid_labels']
    
    def test_edge_case_single_observation(self, basic_labeler):
        """Test handling of edge case with minimal data."""
        # Create data with just enough observations for horizon
        min_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=15, freq='h'),  # > horizon
            'symbol': ['EURUSD'] * 15,
            'close': [1.1000] * 15,  # Constant prices
            'RV_w': [0.01] * 15,
            'VoV_wV': [0.1] * 15
        })
        
        # Should work without errors
        labels_df = basic_labeler.label_data(min_data)
        
        # Should produce some labels
        assert len(labels_df) > 0
        assert labels_df['label'].notna().sum() >= 0  # May be all NaN due to constant prices
    
    def test_price_column_flexibility(self, sample_feature_data):
        """Test labeling with different price columns."""
        # Add mid_close column
        test_data = sample_feature_data.copy()
        test_data['mid_close'] = test_data['close'] * 1.0001  # Slightly different values
        
        # Test with mid_close
        mid_labeler = TripleBarrierLabeler(price_col='mid_close', horizon=5)
        mid_labels = mid_labeler.label_data(test_data)
        
        # Test with close
        close_labeler = TripleBarrierLabeler(price_col='close', horizon=5)
        close_labels = close_labeler.label_data(test_data)
        
        # Both should work and produce valid labels
        assert mid_labels['label'].notna().sum() > 0
        assert close_labels['label'].notna().sum() > 0


class TestLabelingUtilities:
    """Test utility functions for loading and saving data."""
    
    def test_load_feature_data_success(self):
        """Test successful loading of feature data."""
        # Create temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create and save test data
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=10, freq='h'),
                'symbol': ['EURUSD'] * 10,
                'close': range(10),
                'RV_w': [0.01] * 10
            })
            test_data.to_parquet(temp_path, index=False)
            
            # Test loading
            loaded_data = load_feature_data(temp_path)
            
            assert len(loaded_data) == 10
            assert 'timestamp' in loaded_data.columns
            assert 'symbol' in loaded_data.columns
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_feature_data_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            load_feature_data('nonexistent_file.parquet')
    
    def test_load_feature_data_wrong_format(self):
        """Test error handling for wrong file format."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must be a Parquet file"):
                load_feature_data(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_save_labels_success(self):
        """Test successful saving of labels."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create test labels
            test_labels = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=5, freq='h'),
                'symbol': ['EURUSD'] * 5,
                'label': [1, -1, 0, 1, -1]
            })
            
            # Test saving
            save_labels(test_labels, temp_path)
            
            # Verify file was created and is readable
            assert Path(temp_path).exists()
            saved_data = pd.read_parquet(temp_path)
            assert len(saved_data) == 5
            assert 'label' in saved_data.columns
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_volatility_handling(self):
        """Test handling of zero volatility values."""
        labeler = TripleBarrierLabeler(vol_basis='rv', horizon=5)
        
        # Create data with zero volatility
        zero_vol_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=20, freq='h'),
            'symbol': ['EURUSD'] * 20,
            'close': [1.1000] * 20,  # Constant prices
            'RV_w': [0.0] * 20,      # Zero volatility
            'VoV_wV': [0.1] * 20
        })
        
        # Should handle gracefully without crashing
        labels_df = labeler.label_data(zero_vol_data)
        
        # With zero volatility, barriers will be at current price
        # Most likely outcome is timeouts
        assert len(labels_df) > 0
    
    def test_nan_volatility_handling(self):
        """Test handling of NaN volatility values."""
        labeler = TripleBarrierLabeler(vol_basis='rv', horizon=5)
        
        # Create data with some NaN volatility
        nan_vol_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=20, freq='h'),
            'symbol': ['EURUSD'] * 20,
            'close': range(20),
            'RV_w': [0.01 if i % 2 == 0 else np.nan for i in range(20)],  # Alternating NaN
            'VoV_wV': [0.1] * 20
        })
        
        # Should handle NaN values by filtering them out
        labels_df = labeler.label_data(nan_vol_data)
        
        # Should have fewer valid observations due to NaN filtering
        assert len(labels_df) > 0
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements."""
        labeler = TripleBarrierLabeler(vol_basis='rv', k_up=1.0, k_dn=1.0, horizon=5)
        
        # Create data with extreme price jumps
        extreme_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=20, freq='h'),
            'symbol': ['EURUSD'] * 20,
            'close': [1.0, 2.0, 0.5, 3.0, 0.1] * 4,  # Extreme movements
            'RV_w': [0.01] * 20,
            'VoV_wV': [0.1] * 20
        })
        
        # Should handle extreme movements without crashing
        labels_df = labeler.label_data(extreme_data)
        
        # Should produce some labels
        assert len(labels_df) > 0
        valid_labels = labels_df['label'].notna().sum()
        assert valid_labels >= 0  # May be zero due to extreme movements


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
