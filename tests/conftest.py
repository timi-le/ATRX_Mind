"""
Pytest configuration and fixtures for the FX AI-Quant Trading System.

This module provides common fixtures and configuration for all tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Test environment configuration
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_market_data():
    """Provide mock market data for testing."""
    return {
        "EURUSD": {
            "bid": 1.0850,
            "ask": 1.0852,
            "timestamp": "2024-01-01T12:00:00Z",
            "volume": 1000000,
        },
        "GBPUSD": {
            "bid": 1.2650,
            "ask": 1.2652,
            "timestamp": "2024-01-01T12:00:00Z",
            "volume": 800000,
        },
    }


@pytest.fixture
def mock_config():
    """Provide mock system configuration."""
    from core.config.settings import SystemConfig

    config = SystemConfig()
    config.environment = "test"
    config.debug = True
    config.data.fx_pairs = ["EURUSD", "GBPUSD"]
    config.trading.kelly_multiplier = 0.25
    config.risk.max_daily_drawdown = 0.02

    return config


@pytest.fixture
async def mock_data_provider():
    """Provide a mock data provider for testing."""
    provider = AsyncMock()
    provider.connect.return_value = True
    provider.is_connected.return_value = True
    return provider


@pytest.fixture
async def mock_message_bus():
    """Provide a mock message bus for testing."""
    bus = AsyncMock()
    bus.is_connected.return_value = True
    return bus


@pytest.fixture
def mock_ml_predictor():
    """Provide a mock ML predictor for testing."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "prediction": 0.75,
        "confidence": 0.85,
        "model_name": "test_model",
    }
    return predictor


@pytest.fixture
def mock_strategy():
    """Provide a mock trading strategy for testing."""
    strategy = AsyncMock()
    strategy.get_name.return_value = "test_strategy"
    strategy.generate_signal.return_value = {
        "symbol": "EURUSD",
        "side": "buy",
        "strength": 0.8,
        "confidence": 0.7,
    }
    return strategy


@pytest.fixture
def mock_risk_manager():
    """Provide a mock risk manager for testing."""
    risk_manager = AsyncMock()
    risk_manager.check_pre_trade_risk.return_value = True
    risk_manager.monitor_drawdown.return_value = False  # No drawdown violation
    return risk_manager


@pytest.fixture
def mock_execution_engine():
    """Provide a mock execution engine for testing."""
    engine = AsyncMock()
    engine.submit_order.return_value = "order_12345"
    engine.get_account_balance.return_value = 100000.0
    return engine


@pytest.fixture
def sample_ohlc_data():
    """Provide sample OHLC data for testing."""
    import numpy as np
    import pandas as pd

    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")

    # Generate realistic FX price data
    np.random.seed(42)  # For reproducible tests
    price_base = 1.0850
    price_changes = np.random.normal(0, 0.0001, 100)
    prices = price_base + np.cumsum(price_changes)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices + np.random.uniform(0, 0.0005, 100),
            "low": prices - np.random.uniform(0, 0.0005, 100),
            "close": prices + np.random.uniform(-0.0002, 0.0002, 100),
            "volume": np.random.randint(50000, 200000, 100),
        }
    )

    # Ensure high >= max(open, close) and low <= min(open, close)
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    return data


@pytest.fixture
def sample_features():
    """Provide sample feature data for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="1H")

    return pd.DataFrame(
        {
            "timestamp": dates,
            "rsi": np.random.uniform(20, 80, 50),
            "macd": np.random.uniform(-0.001, 0.001, 50),
            "bollinger_upper": np.random.uniform(1.086, 1.088, 50),
            "bollinger_lower": np.random.uniform(1.082, 1.084, 50),
            "volatility": np.random.uniform(0.0001, 0.001, 50),
            "momentum": np.random.uniform(-0.0005, 0.0005, 50),
        }
    )


# Pytest markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


# Custom assertions for trading-specific tests
def assert_valid_order(order):
    """Assert that an order object is valid."""
    required_fields = ["order_id", "symbol", "side", "order_type", "quantity"]
    for field in required_fields:
        assert hasattr(order, field), f"Order missing required field: {field}"

    assert order.quantity > 0, "Order quantity must be positive"
    assert order.side in ["buy", "sell"], "Order side must be 'buy' or 'sell'"


def assert_valid_signal(signal):
    """Assert that a trading signal is valid."""
    required_fields = ["symbol", "side", "strength", "confidence"]
    for field in required_fields:
        assert hasattr(signal, field), f"Signal missing required field: {field}"

    assert 0 <= signal.strength <= 1, "Signal strength must be between 0 and 1"
    assert 0 <= signal.confidence <= 1, "Signal confidence must be between 0 and 1"


def assert_valid_prediction(prediction):
    """Assert that an ML prediction is valid."""
    required_fields = ["prediction", "confidence", "model_name"]
    for field in required_fields:
        assert hasattr(prediction, field), f"Prediction missing required field: {field}"

    assert (
        0 <= prediction.confidence <= 1
    ), "Prediction confidence must be between 0 and 1"
