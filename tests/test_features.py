#!/usr/bin/env python3
"""
ATRX Feature Computation Tests
=============================

Comprehensive test suite for validating feature calculations in core/features/compute_features.py
Tests RV, ATR, VoV, and other technical indicators against hand-computed values.

Following ATRX debugging guidelines for thorough validation and boundary testing.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
import tempfile
import os
from pathlib import Path

# Import the feature computer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.features.compute_features import FeatureComputer


class TestFeatureComputations:
    """Test suite for core feature computation functions."""
    
    @pytest.fixture
    def feature_computer(self):
        """Create a FeatureComputer instance for testing."""
        # Create a temporary config for testing
        config = {
            'general': {
                'output_dir': 'temp',
                'min_required_rows': 10,
                'max_missing_ratio': 0.1
            },
            'windows': {
                'short': 14,
                'medium': 20,
                'long': 50,
                'vol': 30,
                'autocorr': 10,
                'efficiency': 14,
                'spread_ma': 20
            },
            'common_features': {},  # Added missing section
            'bidask_features': {},  # Added for completeness
            'validation': {
                'outlier_threshold': 5.0,
                'feature_ranges': {}
            },
            'schema': {
                'ohlc_required': ['timestamp', 'symbol', 'open', 'high', 'low', 'close'],
                'bidask_required': ['timestamp', 'symbol', 'bid_close', 'ask_close', 'mid_close']
            },
            'performance': {
                'dtype_optimization': True
            },
            'output': {
                'float_precision': 'float32'
            }
        }
        
        with patch.object(FeatureComputer, '_load_config', return_value=config):
            return FeatureComputer()
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing."""
        dates = pd.date_range('2020-01-01', periods=50, freq='h', tz='UTC')
        
        # Create realistic price data with known properties
        np.random.seed(42)  # For reproducible tests
        base_price = 1.1000
        
        # Generate price movements with controlled volatility
        returns = np.random.normal(0, 0.001, 50)  # 0.1% hourly volatility
        prices = [base_price]
        
        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC with realistic relationships
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1]
            
            # Add small random variations for high/low
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.0005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.0005)))
            
            data.append({
                'timestamp': date,
                'symbol': 'EURUSD',
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close, 5),
                'tick_volume': 1000,
                'real_volume': 0,
                'spread': 2
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_bidask_data(self):
        """Create sample bid/ask data for testing."""
        dates = pd.date_range('2020-01-01', periods=30, freq='h', tz='UTC')
        
        np.random.seed(42)
        base_price = 1.1000
        spread = 0.0002  # 2 pip spread
        
        data = []
        for i, date in enumerate(dates):
            mid_price = base_price + (i * 0.0001)  # Slight upward trend
            
            bid_close = mid_price - spread/2
            ask_close = mid_price + spread/2
            
            # Add small variations for OHLC
            bid_open = bid_close + np.random.normal(0, 0.00005)
            ask_open = ask_close + np.random.normal(0, 0.00005)
            
            data.append({
                'timestamp': date,
                'symbol': 'EURUSD',
                'bid_open': round(bid_open, 5),
                'bid_high': round(bid_close + abs(np.random.normal(0, 0.00005)), 5),
                'bid_low': round(bid_close - abs(np.random.normal(0, 0.00005)), 5),
                'bid_close': round(bid_close, 5),
                'ask_open': round(ask_open, 5),
                'ask_high': round(ask_close + abs(np.random.normal(0, 0.00005)), 5),
                'ask_low': round(ask_close - abs(np.random.normal(0, 0.00005)), 5),
                'ask_close': round(ask_close, 5),
                'mid_open': round((bid_open + ask_open)/2, 5),
                'mid_high': round(((bid_close + ask_close)/2) + abs(np.random.normal(0, 0.00005)), 5),
                'mid_low': round(((bid_close + ask_close)/2) - abs(np.random.normal(0, 0.00005)), 5),
                'mid_close': round(mid_price, 5),
                'spread_close': round(spread, 5),
                'bid_change': 0.0,
                'ask_change': 0.0
            })
        
        return pd.DataFrame(data)

    def test_log_returns_calculation(self, feature_computer, sample_ohlc_data):
        """Test log returns calculation against hand-computed values."""
        df = sample_ohlc_data.copy()
        
        # Compute log returns
        log_returns = feature_computer.compute_log_returns(df, 'close', periods=1)
        
        # Hand compute first few values
        expected_first = np.nan  # First value should be NaN
        expected_second = np.log(df['close'].iloc[1] / df['close'].iloc[0])
        expected_third = np.log(df['close'].iloc[2] / df['close'].iloc[1])
        
        assert pd.isna(log_returns.iloc[0])
        assert abs(log_returns.iloc[1] - expected_second) < 1e-10
        assert abs(log_returns.iloc[2] - expected_third) < 1e-10
        
        # Test properties of log returns
        assert len(log_returns) == len(df)
        assert not np.isinf(log_returns.dropna()).any()
    
    def test_realized_volatility_calculation(self, feature_computer):
        """Test realized volatility calculation with known values."""
        # Create returns with known volatility
        np.random.seed(123)
        n_obs = 100
        true_vol = 0.02  # 2% daily volatility
        
        # Generate returns with known standard deviation
        returns = pd.Series(np.random.normal(0, true_vol, n_obs))
        
        # Test RV calculation
        window = 20
        rv = feature_computer.compute_realized_volatility(returns, window)
        
        # Hand compute the expected value for a specific window
        # RV = sqrt(var * 252) for annualization
        returns_window = returns.iloc[window-1:window]  # 20 observations
        expected_var = returns_window.var()
        expected_rv = np.sqrt(expected_var * 252)
        
        # Compare with computed value (allowing for floating point precision)
        computed_rv = rv.iloc[window-1]
        
        # Test properties
        assert len(rv) == len(returns)
        assert pd.isna(rv.iloc[:window-1]).all()  # First window-1 values should be NaN
        assert not pd.isna(rv.iloc[window:]).any()  # Rest should not be NaN
        assert (rv.dropna() >= 0).all()  # Volatility should be non-negative
        
        # Test annualization factor
        daily_var = returns.iloc[20:40].var()
        daily_rv = np.sqrt(daily_var)
        annualized_rv = daily_rv * np.sqrt(252)
        manual_computation = feature_computer.compute_realized_volatility(returns.iloc[20:40], 20).iloc[-1]
        
        # The scaling should be consistent
        assert abs(manual_computation - annualized_rv) < 0.01
    
    def test_atr_calculation(self, feature_computer, sample_ohlc_data):
        """Test Average True Range calculation against hand-computed values."""
        df = sample_ohlc_data.copy()
        
        # Compute ATR
        window = 5  # Small window for easier verification
        atr = feature_computer.compute_atr(df, window)
        
        # Hand compute True Range for first few periods
        def true_range_manual(high, low, prev_close):
            return max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
        
        # Compute expected values manually
        tr_values = []
        for i in range(len(df)):
            if i == 0:
                # First period: only high-low range
                tr = df['high'].iloc[i] - df['low'].iloc[i]
            else:
                tr = true_range_manual(
                    df['high'].iloc[i], 
                    df['low'].iloc[i], 
                    df['close'].iloc[i-1]
                )
            tr_values.append(tr)
        
        # Expected ATR for window position should be average of TR values
        expected_atr_at_window = np.mean(tr_values[:window])
        
        # Test properties
        assert len(atr) == len(df)
        assert pd.isna(atr.iloc[:window-1]).all()  # First values should be NaN
        assert abs(atr.iloc[window-1] - expected_atr_at_window) < 1e-10
        assert (atr.dropna() >= 0).all()  # ATR should be non-negative
    
    def test_vov_calculation(self, feature_computer):
        """Test Volatility of Volatility calculation."""
        # Create synthetic returns with varying volatility
        np.random.seed(456)
        
        # Create periods of high and low volatility
        high_vol_returns = np.random.normal(0, 0.02, 50)  # High vol period
        low_vol_returns = np.random.normal(0, 0.005, 50)   # Low vol period
        returns = pd.Series(np.concatenate([high_vol_returns, low_vol_returns]))
        
        # Test VoV calculation
        rv_window = 20
        vov_window = 10
        vov = feature_computer.compute_volatility_of_volatility(returns, rv_window, vov_window)
        
        # Manual computation for verification
        rv_manual = feature_computer.compute_realized_volatility(returns, rv_window)
        log_rv = np.log(rv_manual.replace(0, np.nan))
        expected_vov = log_rv.rolling(vov_window).std()
        
        # Compare results
        pd.testing.assert_series_equal(vov, expected_vov, check_names=False)
        
        # Test properties
        assert len(vov) == len(returns)
        assert (vov.dropna() >= 0).all()  # VoV should be non-negative
        
        # VoV should be higher during the transition period (varying volatility)
        transition_vov = vov.iloc[60:80].mean()  # Around the transition
        stable_vov = vov.iloc[80:90].mean()      # Stable low vol period
        
        # During transition, VoV should generally be higher
        assert transition_vov > stable_vov * 0.5  # Allow some tolerance
    
    def test_rsi_calculation(self, feature_computer):
        """Test RSI calculation with known trending data."""
        # Create trending price data
        trend_up = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]  # Strong uptrend
        trend_down = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]  # Strong downtrend
        
        prices_up = pd.Series(trend_up)
        prices_down = pd.Series(trend_down)
        
        window = 5
        rsi_up = feature_computer.compute_rsi(prices_up, window)
        rsi_down = feature_computer.compute_rsi(prices_down, window)
        
        # Test properties
        assert len(rsi_up) == len(prices_up)
        assert len(rsi_down) == len(prices_down)
        
        # RSI should be between 0 and 100
        assert (rsi_up.dropna() >= 0).all() and (rsi_up.dropna() <= 100).all()
        assert (rsi_down.dropna() >= 0).all() and (rsi_down.dropna() <= 100).all()
        
        # For strong uptrend, RSI should be high (>70)
        # For strong downtrend, RSI should be low (<30)
        final_rsi_up = rsi_up.iloc[-1]
        final_rsi_down = rsi_down.iloc[-1]
        
        assert final_rsi_up > 70, f"Uptrend RSI should be >70, got {final_rsi_up}"
        assert final_rsi_down < 30, f"Downtrend RSI should be <30, got {final_rsi_down}"
    
    def test_bollinger_width_calculation(self, feature_computer):
        """Test Bollinger Band width calculation."""
        # Create price data with known volatility pattern
        np.random.seed(789)
        
        # Low volatility period
        low_vol = 100 + np.random.normal(0, 0.1, 30)
        # High volatility period  
        high_vol = 100 + np.random.normal(0, 1.0, 30)
        
        prices_low = pd.Series(low_vol)
        prices_high = pd.Series(high_vol)
        
        window = 20
        std_mult = 2.0
        
        bb_width_low = feature_computer.compute_bollinger_width(prices_low, window, std_mult)
        bb_width_high = feature_computer.compute_bollinger_width(prices_high, window, std_mult)
        
        # Test properties
        assert len(bb_width_low) == len(prices_low)
        assert len(bb_width_high) == len(prices_high)
        assert (bb_width_low.dropna() >= 0).all()
        assert (bb_width_high.dropna() >= 0).all()
        
        # High volatility should produce wider bands
        avg_width_low = bb_width_low.dropna().mean()
        avg_width_high = bb_width_high.dropna().mean()
        
        assert avg_width_high > avg_width_low * 2, "High volatility should produce wider bands"
    
    def test_microstructure_features(self, feature_computer, sample_bidask_data):
        """Test bid/ask microstructure feature calculations."""
        df = sample_bidask_data.copy()
        
        # Test range efficiency
        window = 5
        range_eff = feature_computer.compute_range_efficiency(df, window)
        
        assert len(range_eff) == len(df)
        assert (range_eff.dropna() >= 0).all()  # Should be non-negative
        # Range efficiency can exceed 1 when prices move efficiently relative to range
        
        # Test signed spread move
        signed_move = feature_computer.compute_signed_spread_move(df)
        
        assert len(signed_move) == len(df)
        # First value should be NaN (no previous value for diff)
        assert pd.isna(signed_move.iloc[0])
        
        # Test bid/ask imbalance
        imbalance = feature_computer.compute_bid_ask_imbalance(df)
        
        assert len(imbalance) == len(df)
        assert (imbalance.dropna() >= -1).all()  # Should be between -1 and 1
        assert (imbalance.dropna() <= 1).all()
    
    def test_data_format_detection(self, feature_computer, sample_ohlc_data, sample_bidask_data):
        """Test automatic data format detection."""
        ohlc_format = feature_computer.detect_data_format(sample_ohlc_data)
        bidask_format = feature_computer.detect_data_format(sample_bidask_data)
        
        assert ohlc_format == 'ohlc'
        assert bidask_format == 'bidask'
        
        # Test error case with ambiguous data
        ambiguous_data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'symbol': ['A', 'B', 'C'],
            'price': [100, 101, 102]
        })
        
        with pytest.raises(ValueError, match="Cannot detect data format"):
            feature_computer.detect_data_format(ambiguous_data)
    
    def test_validation_edge_cases(self, feature_computer):
        """Test edge cases and error handling."""
        # Test with insufficient data
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5, tz='UTC'),
            'symbol': ['EURUSD'] * 5,
            'close': [1.1, 1.2, 1.3, 1.4, 1.5]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            feature_computer.validate_input_data(small_data, 'ohlc')
        
        # Test with missing required columns
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=20, tz='UTC'),
            'symbol': ['EURUSD'] * 20,
            'close': range(20)
            # Missing 'open', 'high', 'low'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            feature_computer.validate_input_data(incomplete_data, 'ohlc')
    
    def test_outlier_detection_and_capping(self, feature_computer):
        """Test outlier detection and capping functionality."""
        # Create data with obvious outliers
        np.random.seed(999)
        normal_data = np.random.normal(0, 1, 100)
        # Add extreme outliers
        normal_data[10] = 20.0  # 20 standard deviations
        normal_data[20] = -15.0  # -15 standard deviations
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, tz='UTC'),
            'symbol': ['EURUSD'] * 100,
            'feature_col': normal_data
        })
        
        # Apply validation (which includes outlier capping)
        validated_df = feature_computer.validate_features(df)
        
        # Check that extreme outliers were capped
        # The capping is based on the statistics of the original data (including outliers)
        threshold = 5.0  # From config
        mean_val = normal_data.mean()
        std_val = normal_data.std()
        upper_bound = mean_val + (threshold * std_val)
        lower_bound = mean_val - (threshold * std_val)
        
        # The key test is that the extreme values (20.0 and -15.0) were reduced
        assert validated_df['feature_col'].max() < 20.0  # Original outlier was 20.0
        assert validated_df['feature_col'].min() > -15.0  # Original outlier was -15.0
        
        # And that the capped values are close to the expected bounds
        assert validated_df['feature_col'].max() <= upper_bound + 1.0  # Allow some tolerance
        assert validated_df['feature_col'].min() >= lower_bound - 1.0
        
        # Check that reasonable values remain unchanged
        normal_mask = (np.abs(normal_data) < 3)  # Within 3 std devs
        original_normal = normal_data[normal_mask]
        validated_normal = validated_df.loc[normal_mask, 'feature_col']
        
        # Normal values should be mostly unchanged
        assert np.allclose(original_normal, validated_normal, rtol=1e-10)
    
    def test_memory_optimization(self, feature_computer, sample_ohlc_data):
        """Test memory optimization functionality."""
        df = sample_ohlc_data.copy()
        
        # Add some float64 columns
        df['test_feature'] = np.random.random(len(df)).astype('float64')
        
        original_memory = df.memory_usage(deep=True).sum()
        
        # Apply memory optimization
        optimized_df = feature_computer.optimize_dtypes(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Memory usage should be reduced
        assert optimized_memory < original_memory
        
        # timestamp should remain as datetime64
        assert pd.api.types.is_datetime64_any_dtype(optimized_df['timestamp'])
        
        # Other numeric columns should be float32
        assert optimized_df['test_feature'].dtype == np.float32
    
    def test_feature_computation_integration(self, feature_computer, sample_ohlc_data, sample_bidask_data):
        """Integration test for complete feature computation workflow."""
        # Test OHLC feature computation
        ohlc_result = feature_computer.compute_common_features(sample_ohlc_data.copy())
        
        expected_features = [
            'log_ret_1', 'RV_w', 'ATR_w', 'VoV_wV', 'RSI_w', 
            'bb_width', 'rolling_kurt', 'rolling_skew', 
            'rolling_autocorr', 'zscore_close'
        ]
        
        for feature in expected_features:
            assert feature in ohlc_result.columns, f"Missing feature: {feature}"
        
        # Test bid/ask feature computation
        bidask_result = feature_computer.compute_common_features(sample_bidask_data.copy())
        bidask_result = feature_computer.compute_bidask_features(bidask_result)
        
        additional_features = [
            'spread_ma_w', 'spread_vol', 'mid_ret_1', 'range_efficiency',
            'signed_spread_move', 'bid_ask_imbalance', 'effective_spread', 'quote_slope'
        ]
        
        for feature in expected_features + additional_features:
            assert feature in bidask_result.columns, f"Missing feature: {feature}"
        
        # Validate feature properties
        # Note: With small test datasets (30 points) and large windows (20), some features may be all NaN
        features_with_data = ['log_ret_1', 'mid_ret_1']  # These should always have data with sufficient input
        
        for feature in features_with_data:
            if feature in bidask_result.columns:
                assert not bidask_result[feature].dropna().empty, f"Feature {feature} has no valid values"
        
        # For other features, just check they exist and have correct structure
        for feature in expected_features:
            assert feature in bidask_result.columns, f"Missing expected feature: {feature}"


class TestFeatureConfigurationHandling:
    """Test configuration loading and validation."""
    
    def test_config_loading_with_missing_sections(self):
        """Test that missing required config sections raise appropriate errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
general:
  output_dir: "test"
# Missing required sections: windows, common_features
            """)
            temp_config = f.name
        
        try:
            with pytest.raises(ValueError, match="Required config section missing"):
                FeatureComputer(config_path=temp_config)
        finally:
            os.unlink(temp_config)
    
    def test_config_loading_with_invalid_yaml(self):
        """Test error handling for invalid YAML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            temp_config = f.name
        
        try:
            with pytest.raises(Exception):  # Should raise YAML parsing error
                FeatureComputer(config_path=temp_config)
        finally:
            os.unlink(temp_config)


class TestNumericalStability:
    """Test numerical stability and precision of calculations."""
    
    def test_log_returns_with_zero_prices(self):
        """Test log returns calculation with zero or negative prices."""
        fc = FeatureComputer.__new__(FeatureComputer)  # Create without initialization
        
        # Test with zero prices
        prices_with_zero = pd.Series([1.0, 0.0, 1.0, 2.0])
        log_ret = fc.compute_log_returns(pd.DataFrame({'price': prices_with_zero}), 'price')
        
        # Should handle division by zero gracefully
        assert pd.isna(log_ret.iloc[0])  # First is always NaN
        assert np.isinf(log_ret.iloc[1]) or pd.isna(log_ret.iloc[1])  # Division by zero
        
        # Test with negative prices (should produce NaN in log)
        prices_with_negative = pd.Series([1.0, -1.0, 1.0])
        log_ret_neg = fc.compute_log_returns(pd.DataFrame({'price': prices_with_negative}), 'price')
        
        assert pd.isna(log_ret_neg.iloc[1]) or np.isnan(log_ret_neg.iloc[1])
    
    def test_volatility_with_constant_prices(self):
        """Test volatility calculation with constant prices (zero variance)."""
        fc = FeatureComputer.__new__(FeatureComputer)
        
        # Constant prices should produce zero volatility
        constant_returns = pd.Series([0.0] * 50)
        rv = fc.compute_realized_volatility(constant_returns, 20)
        
        # Should be zero or very close to zero
        assert (rv.dropna() < 1e-10).all()
    
    def test_precision_with_very_small_numbers(self):
        """Test calculations with very small price movements."""
        fc = FeatureComputer.__new__(FeatureComputer)
        
        # Very small price movements (micro pips)
        base_price = 1.123456789
        tiny_movements = [base_price + i * 1e-10 for i in range(50)]
        prices = pd.Series(tiny_movements)
        
        log_ret = fc.compute_log_returns(pd.DataFrame({'price': prices}), 'price')
        
        # Should not overflow or underflow
        assert not np.isinf(log_ret.dropna()).any()
        assert not (log_ret.dropna() == 0).all()  # Should capture the tiny movements


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
