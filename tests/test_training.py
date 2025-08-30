"""
ATRX Training System Tests
=========================

Comprehensive tests for LSTM, CNN, and XGBoost training scripts.
Validates model creation, training workflow, and output generation.

Following ATRX development standards for thorough ML testing.
"""

import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import yaml

# Test data constants
TEST_SAMPLES = 1000
TEST_FEATURES = 10
TEST_SEQUENCE_LENGTH = 32


class TestTrainingBase:
    """Base class for training tests with shared utilities."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_dataset(self, temp_dir):
        """Create test dataset for training."""
        np.random.seed(42)
        
        # Generate synthetic time series data
        timestamps = pd.date_range('2020-01-01', periods=TEST_SAMPLES, freq='H')
        
        # Create realistic financial features
        price_base = 1.2000  # EUR/USD base price
        returns = np.random.normal(0, 0.001, TEST_SAMPLES)  # Realistic FX returns
        prices = [price_base]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = {
            'timestamp': timestamps,
            'symbol': ['EURUSD'] * TEST_SAMPLES,
            'close': prices,
        }
        
        # Add technical indicators
        for i in range(TEST_FEATURES):
            feature_name = f'feature_{i}'
            # Create correlated features with some noise
            base_signal = np.sin(np.linspace(0, 4*np.pi, TEST_SAMPLES)) * (i + 1)
            noise = np.random.normal(0, 0.1, TEST_SAMPLES)
            data[feature_name] = base_signal + noise
        
        # Add realistic labels (Triple Barrier style)
        labels = np.random.choice([-1, 0, 1], TEST_SAMPLES, p=[0.3, 0.4, 0.3])
        # Encode to {0, 1, 2}
        data['label'] = [0 if x == -1 else 1 if x == 0 else 2 for x in labels]
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        dataset_path = temp_dir / "test_dataset.parquet"
        df.to_parquet(dataset_path, index=False)
        
        return dataset_path
    
    @pytest.fixture
    def test_splits(self, temp_dir):
        """Create test time series splits."""
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Create 3 simple splits for testing
        splits = []
        split_size = TEST_SAMPLES // 4
        
        for i in range(3):
            train_start = i * split_size
            train_end = train_start + 2 * split_size
            val_start = train_end
            val_end = val_start + split_size
            
            # Ensure we don't exceed bounds
            val_end = min(val_end, TEST_SAMPLES)
            
            if val_end <= val_start:
                break
            
            split = {
                'split_id': i,
                'splits': {
                    'EURUSD': {
                        'train_indices': list(range(train_start, train_end)),
                        'val_indices': list(range(val_start, val_end)),
                        'train_samples': train_end - train_start,
                        'val_samples': val_end - val_start
                    }
                }
            }
            splits.append(split)
        
        # Save splits file
        splits_file = splits_dir / "test_splits.json"
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        return splits_dir
    
    @pytest.fixture
    def features_config(self, temp_dir):
        """Create test features configuration."""
        config = {
            'windows': {
                'short': 10,
                'medium': 20,
                'long': 50
            },
            'validation': {
                'min_periods': 5,
                'outlier_threshold': 3.0
            }
        }
        
        config_path = temp_dir / "features.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def training_config(self, temp_dir):
        """Create test training configuration."""
        config = {
            'global': {
                'seed': 42,
                'early_stopping_patience': 3,
                'test_mode': True  # Fast training for tests
            },
            'lstm': {
                'sequence_length': TEST_SEQUENCE_LENGTH,
                'lstm_units': [32, 16],
                'dense_units': [16],
                'epochs': 5,  # Fast for testing
                'batch_size': 64,
                'learning_rate': 0.01
            },
            'cnn': {
                'sequence_length': TEST_SEQUENCE_LENGTH,
                'conv_filters': [16, 32],
                'kernel_sizes': [3, 3],
                'pool_sizes': [1, 1],  # No pooling for small sequences
                'dense_units': [16],
                'epochs': 5,  # Fast for testing
                'batch_size': 64,
                'learning_rate': 0.01
            },
            'xgboost': {
                'n_estimators': 50,  # Fast for testing
                'max_depth': 3,
                'learning_rate': 0.3,
                'early_stopping_rounds': 5
            },
            'feature_engineering': {
                'scale_features': True,
                'scaler_type': 'standard',
                'encode_labels': True
            }
        }
        
        config_path = temp_dir / "training.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def check_training_outputs(self, output_dir: Path, model_type: str, num_folds: int):
        """Check that training produced expected outputs."""
        output_dir = Path(output_dir)
        
        # Check main output directory exists
        assert output_dir.exists(), f"Output directory not created: {output_dir}"
        
        # Check fold directories
        for fold_id in range(num_folds):
            fold_dir = output_dir / f"fold_{fold_id}"
            assert fold_dir.exists(), f"Fold directory not created: {fold_dir}"
            
            # Check fold files
            assert (fold_dir / "metrics.json").exists(), f"Metrics not saved for fold {fold_id}"
            assert (fold_dir / "predictions.parquet").exists(), f"Predictions not saved for fold {fold_id}"
            
            # Check model files (format depends on model type)
            if model_type in ['lstm', 'cnn']:
                assert (fold_dir / "model.h5").exists(), f"Keras model not saved for fold {fold_id}"
                assert (fold_dir / "saved_model").exists(), f"SavedModel not saved for fold {fold_id}"
            elif model_type == 'xgboost':
                assert (fold_dir / "model.xgb").exists(), f"XGBoost model not saved for fold {fold_id}"
                assert (fold_dir / "model.pkl").exists(), f"Pickle model not saved for fold {fold_id}"
        
        # Check aggregated outputs
        assert (output_dir / "oof_predictions.parquet").exists(), "OOF predictions not saved"
        assert (output_dir / "training_summary.json").exists(), "Training summary not saved"
        assert (output_dir / "feature_names.json").exists(), "Feature names not saved"
        assert (output_dir / "scalers.pkl").exists(), "Scalers not saved"
    
    def check_oof_predictions(self, output_dir: Path, expected_length: int):
        """Check out-of-fold predictions quality."""
        oof_path = output_dir / "oof_predictions.parquet"
        assert oof_path.exists(), "OOF predictions file not found"
        
        oof_df = pd.read_parquet(oof_path)
        
        # Check structure
        assert len(oof_df) == expected_length, f"OOF length mismatch: {len(oof_df)} != {expected_length}"
        assert 'oof_prediction' in oof_df.columns, "OOF prediction column missing"
        
        # Check coverage (should have some valid predictions)
        valid_predictions = (oof_df['oof_prediction'] != -999).sum()
        coverage = valid_predictions / expected_length
        assert coverage > 0.5, f"Low OOF coverage: {coverage:.1%}"
        
        # Check prediction values are valid
        valid_preds = oof_df[oof_df['oof_prediction'] != -999]['oof_prediction']
        assert valid_preds.min() >= 0, "Invalid negative predictions"
        assert valid_preds.max() <= 2, "Invalid predictions > 2"


class TestLSTMTraining(TestTrainingBase):
    """Test LSTM training functionality."""
    
    def test_lstm_training_workflow(self, temp_dir, test_dataset, test_splits, 
                                  features_config, training_config):
        """Test complete LSTM training workflow."""
        from trainers.train_lstm import LSTMTrainer
        
        output_dir = temp_dir / "lstm_output"
        
        # Initialize trainer
        trainer = LSTMTrainer(
            dataset_path=str(test_dataset),
            splits_dir=str(test_splits),
            features_config_path=str(features_config),
            training_config_path=str(training_config),
            output_dir=str(output_dir),
            sequence_length=TEST_SEQUENCE_LENGTH
        )
        
        # Run training
        trainer.run_training()
        
        # Check outputs
        self.check_training_outputs(output_dir, 'lstm', num_folds=3)
        self.check_oof_predictions(output_dir, TEST_SAMPLES)
    
    def test_lstm_sequence_creation(self, temp_dir, test_dataset, test_splits,
                                  features_config, training_config):
        """Test LSTM sequence creation logic."""
        from trainers.train_lstm import LSTMTrainer
        
        trainer = LSTMTrainer(
            dataset_path=str(test_dataset),
            splits_dir=str(test_splits),
            features_config_path=str(features_config),
            training_config_path=str(training_config),
            output_dir=str(temp_dir / "lstm_test"),
            sequence_length=TEST_SEQUENCE_LENGTH
        )
        
        # Load data and prepare features
        df = trainer.load_data()
        X = trainer.prepare_features(df.iloc[:100], fit_scaler=True)
        y = trainer.prepare_labels(df.iloc[:100]['label'])
        
        # Create sequences
        X_seq, y_seq = trainer.create_sequences(X, y)
        
        # Verify sequence shapes
        expected_samples = len(X) - TEST_SEQUENCE_LENGTH + 1
        assert X_seq.shape == (expected_samples, TEST_SEQUENCE_LENGTH, X.shape[1])
        assert y_seq.shape == (expected_samples,)
        
        # Verify temporal alignment
        # Last sequence should use the last TEST_SEQUENCE_LENGTH samples of X
        # and predict the last label
        np.testing.assert_array_equal(X_seq[-1], X[-TEST_SEQUENCE_LENGTH:])
        assert y_seq[-1] == y[-1]


class TestCNNTraining(TestTrainingBase):
    """Test CNN training functionality."""
    
    def test_cnn_training_workflow(self, temp_dir, test_dataset, test_splits,
                                 features_config, training_config):
        """Test complete CNN training workflow."""
        from trainers.train_cnn import CNNTrainer
        
        output_dir = temp_dir / "cnn_output"
        
        # Initialize trainer
        trainer = CNNTrainer(
            dataset_path=str(test_dataset),
            splits_dir=str(test_splits),
            features_config_path=str(features_config),
            training_config_path=str(training_config),
            output_dir=str(output_dir),
            sequence_length=TEST_SEQUENCE_LENGTH
        )
        
        # Run training
        trainer.run_training()
        
        # Check outputs
        self.check_training_outputs(output_dir, 'cnn', num_folds=3)
        self.check_oof_predictions(output_dir, TEST_SAMPLES)
    
    def test_cnn_model_architecture(self, temp_dir, test_dataset, test_splits,
                                   features_config, training_config):
        """Test CNN model architecture creation."""
        from trainers.train_cnn import CNNTrainer
        
        trainer = CNNTrainer(
            dataset_path=str(test_dataset),
            splits_dir=str(test_splits),
            features_config_path=str(features_config),
            training_config_path=str(training_config),
            output_dir=str(temp_dir / "cnn_test"),
            sequence_length=TEST_SEQUENCE_LENGTH
        )
        
        # Load data to get feature count
        df = trainer.load_data()
        input_shape = (TEST_SEQUENCE_LENGTH, len(trainer.feature_names))
        
        # Build model
        model = trainer.build_model(input_shape)
        
        # Check model structure
        assert model.input_shape[1:] == input_shape
        assert model.output_shape == (None, 3)  # 3 classes
        assert model.count_params() > 0


class TestXGBoostTraining(TestTrainingBase):
    """Test XGBoost training functionality."""
    
    def test_xgboost_training_workflow(self, temp_dir, test_dataset, test_splits,
                                     features_config, training_config):
        """Test complete XGBoost training workflow."""
        from trainers.train_xgboost import XGBoostTrainer
        
        output_dir = temp_dir / "xgboost_output"
        
        # Initialize trainer
        trainer = XGBoostTrainer(
            dataset_path=str(test_dataset),
            splits_dir=str(test_splits),
            features_config_path=str(features_config),
            training_config_path=str(training_config),
            output_dir=str(output_dir)
        )
        
        # Run training
        trainer.run_training()
        
        # Check outputs
        self.check_training_outputs(output_dir, 'xgboost', num_folds=3)
        self.check_oof_predictions(output_dir, TEST_SAMPLES)
        
        # Check XGBoost-specific outputs
        assert (output_dir / "feature_importance_summary.json").exists()
    
    def test_xgboost_feature_importance(self, temp_dir, test_dataset, test_splits,
                                       features_config, training_config):
        """Test XGBoost feature importance calculation."""
        from trainers.train_xgboost import XGBoostTrainer
        
        output_dir = temp_dir / "xgboost_test"
        
        trainer = XGBoostTrainer(
            dataset_path=str(test_dataset),
            splits_dir=str(test_splits),
            features_config_path=str(features_config),
            training_config_path=str(training_config),
            output_dir=str(output_dir)
        )
        
        # Run training
        trainer.run_training()
        
        # Check feature importance files
        for fold_id in range(3):
            fold_dir = output_dir / f"fold_{fold_id}"
            assert (fold_dir / "feature_importance.json").exists()
            assert (fold_dir / "feature_importance.csv").exists()
        
        # Check aggregated importance
        importance_file = output_dir / "feature_importance_summary.json"
        assert importance_file.exists()
        
        with open(importance_file, 'r') as f:
            importance_data = json.load(f)
        
        # Should have different importance types
        assert 'weight' in importance_data
        assert len(importance_data['weight']) > 0


class TestTrainingIntegration:
    """Integration tests for training system."""
    
    def test_training_scripts_cli(self, temp_dir):
        """Test training scripts can be called via CLI (mock test)."""
        # This is a mock test - in practice would test actual CLI calls
        from trainers.train_lstm import main as lstm_main
        from trainers.train_cnn import main as cnn_main  
        from trainers.train_xgboost import main as xgb_main
        
        # Verify main functions exist and are callable
        assert callable(lstm_main)
        assert callable(cnn_main)
        assert callable(xgb_main)
    
    def test_config_validation(self, temp_dir):
        """Test configuration validation and error handling."""
        from trainers.base_trainer import BaseTrainer
        
        # Test with missing config file
        with pytest.raises(FileNotFoundError):
            BaseTrainer(
                dataset_path="nonexistent.parquet",
                splits_dir="nonexistent",
                features_config_path="nonexistent.yaml",
                training_config_path="nonexistent.yaml", 
                output_dir=str(temp_dir),
                model_type="test"
            )
    
    def test_oof_prediction_aggregation(self, temp_dir):
        """Test out-of-fold prediction aggregation logic."""
        from trainers.base_trainer import BaseTrainer
        
        # Create minimal trainer instance for testing
        class TestTrainer(BaseTrainer):
            def _save_model(self, model, path):
                pass
            
            def train_fold(self, X_train, y_train, X_val, y_val):
                return None, np.random.randint(0, 3, len(y_val))
            
            def _predict(self, model, X):
                return np.random.randint(0, 3, len(X))
        
        # Create mock configurations
        features_config = temp_dir / "features.yaml"
        training_config = temp_dir / "training.yaml"
        
        with open(features_config, 'w') as f:
            yaml.dump({}, f)
        
        with open(training_config, 'w') as f:
            yaml.dump({'global': {}, 'test': {}}, f)
        
        # Create mock dataset
        data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='H'),
            'symbol': ['TEST'] * 100,
            'label': np.random.randint(0, 3, 100),
            'feature_1': np.random.randn(100)
        })
        
        dataset_path = temp_dir / "test.parquet"
        data.to_parquet(dataset_path, index=False)
        
        # Create mock splits
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir()
        
        splits = [{
            'split_id': 0,
            'splits': {
                'TEST': {
                    'train_indices': list(range(0, 80)),
                    'val_indices': list(range(80, 100)),
                    'train_samples': 80,
                    'val_samples': 20
                }
            }
        }]
        
        with open(splits_dir / "test_splits.json", 'w') as f:
            json.dump(splits, f)
        
        # Test trainer
        trainer = TestTrainer(
            dataset_path=str(dataset_path),
            splits_dir=str(splits_dir),
            features_config_path=str(features_config),
            training_config_path=str(training_config),
            output_dir=str(temp_dir / "test_output"),
            model_type="test"
        )
        
        # Mock some fold results
        trainer.fold_results = [{'fold_id': 0, 'train_metrics': {}, 'val_metrics': {}}]
        
        # Create mock predictions
        fold_dir = trainer.output_dir / "fold_0"
        fold_dir.mkdir(parents=True)
        
        pred_df = pd.DataFrame({
            'index': list(range(80, 100)),
            'prediction': np.random.randint(0, 3, 20)
        })
        pred_df.to_parquet(fold_dir / "predictions.parquet", index=False)
        
        # Test OOF aggregation
        trainer.save_oof_predictions(100)
        
        # Verify OOF file was created
        oof_path = trainer.output_dir / "oof_predictions.parquet"
        assert oof_path.exists()
        
        oof_df = pd.read_parquet(oof_path)
        assert len(oof_df) == 100
        assert 'oof_prediction' in oof_df.columns
