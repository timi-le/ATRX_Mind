#!/usr/bin/env python3
"""
ATRX LSTM Training Script
========================

Trains LSTM models for financial time series prediction using walk-forward validation.
Implements sequence-based learning with proper temporal structure preservation.

Following ATRX development standards for production-ready ML training.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.models import Sequential
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from trainers.base_trainer import BaseTrainer

# Configure TensorFlow
tf.config.experimental.enable_memory_growth = True
tf.get_logger().setLevel('ERROR')

logger = structlog.get_logger(__name__)


class LSTMTrainer(BaseTrainer):
    """LSTM model trainer for financial time series."""
    
    def __init__(self, *args, sequence_length: int = 64, **kwargs):
        """
        Initialize LSTM trainer.
        
        Args:
            sequence_length: Length of input sequences
        """
        super().__init__(*args, model_type='lstm', **kwargs)
        self.sequence_length = sequence_length or self.model_config.get('sequence_length', 64)
        
        # Enable mixed precision if configured
        if self.training_config.get('hardware', {}).get('mixed_precision', True):
            try:
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled")
            except Exception as e:
                logger.warning("Failed to enable mixed precision", error=str(e))
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            X: Feature array [samples, features]
            y: Label array [samples]
            
        Returns:
            X_seq: Sequence array [samples - seq_len + 1, seq_len, features]
            y_seq: Label array [samples - seq_len + 1]
        """
        if len(X) < self.sequence_length:
            raise ValueError(f"Data length {len(X)} < sequence length {self.sequence_length}")
        
        n_samples = len(X) - self.sequence_length + 1
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.sequence_length]
            y_seq[i] = y[i + self.sequence_length - 1]  # Predict the last label
        
        return X_seq, y_seq
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        config = self.model_config
        
        model = Sequential(name=f'LSTM_{self.sequence_length}')
        
        # Input layer
        model.add(layers.Input(shape=input_shape, name='input'))
        
        # LSTM layers
        lstm_units = config.get('lstm_units', [128, 64])
        dropout_rate = config.get('dropout_rate', 0.3)
        recurrent_dropout = config.get('recurrent_dropout', 0.2)
        
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)  # Return sequences except for last layer
            
            model.add(layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=keras.regularizers.L1L2(
                    l1=config.get('l1_reg', 0.0001),
                    l2=config.get('l2_reg', 0.0001)
                ),
                name=f'lstm_{i+1}'
            ))
            
            if config.get('batch_normalization', True):
                model.add(layers.BatchNormalization(name=f'bn_lstm_{i+1}'))
        
        # Dense head
        dense_units = config.get('dense_units', [32, 16])
        activation = config.get('activation', 'tanh')
        
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(
                units=units,
                activation=activation,
                kernel_regularizer=keras.regularizers.L1L2(
                    l1=config.get('l1_reg', 0.0001),
                    l2=config.get('l2_reg', 0.0001)
                ),
                name=f'dense_{i+1}'
            ))
            
            model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))
            
            if config.get('batch_normalization', True):
                model.add(layers.BatchNormalization(name=f'bn_dense_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(
            units=3,  # Three classes: {0, 1, 2}
            activation=config.get('output_activation', 'softmax'),
            name='output'
        ))
        
        # Compile model
        optimizer_name = config.get('optimizer', 'adam')
        learning_rate = config.get('learning_rate', 0.001)
        
        if optimizer_name.lower() == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name
        
        model.compile(
            optimizer=optimizer,
            loss=config.get('loss', 'sparse_categorical_crossentropy'),
            metrics=config.get('metrics', ['accuracy'])
        )
        
        return model
    
    def get_callbacks(self, fold_id: int) -> list:
        """Create training callbacks."""
        config = self.model_config
        fold_dir = self.output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.global_config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.get('reduce_lr_factor', 0.5),
            patience=config.get('reduce_lr_patience', 5),
            min_lr=config.get('min_lr', 0.00001),
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(fold_dir / "best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        # CSV logger for training history
        csv_logger = callbacks.CSVLogger(
            filename=str(fold_dir / "training_history.csv"),
            append=False
        )
        callback_list.append(csv_logger)
        
        return callback_list
    
    def train_fold(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Train LSTM model for one fold."""
        
        # Create sequences
        logger.info("Creating sequences",
                   sequence_length=self.sequence_length,
                   train_samples=len(X_train),
                   val_samples=len(X_val))
        
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        
        # Store original validation labels for consistent metric calculation
        self._original_y_val = y_val_seq
        
        logger.info("Sequences created",
                   train_sequences=len(X_train_seq),
                   val_sequences=len(X_val_seq),
                   input_shape=X_train_seq.shape[1:])
        
        # Build model
        model = self.build_model(input_shape=X_train_seq.shape[1:])
        
        logger.info("Model built",
                   total_params=model.count_params(),
                   trainable_params=sum([tf.size(w).numpy() for w in model.trainable_weights]))
        
        # Get callbacks
        fold_id = len(self.fold_results)  # Current fold number
        model_callbacks = self.get_callbacks(fold_id)
        
        # Training parameters
        config = self.model_config
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 256)
        
        # Train model
        logger.info("Starting training",
                   epochs=epochs,
                   batch_size=batch_size,
                   train_sequences=len(X_train_seq),
                   val_sequences=len(X_val_seq))
        
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=model_callbacks,
            verbose=1
        )
        
        # Make predictions
        val_pred_proba = model.predict(X_val_seq, batch_size=batch_size)
        val_predictions = np.argmax(val_pred_proba, axis=1)
        
        logger.info("Fold training completed",
                   final_train_loss=history.history['loss'][-1],
                   final_val_loss=history.history['val_loss'][-1],
                   best_epoch=len(history.history['loss']))
        
        return model, val_predictions
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Override to use sequenced validation labels when available."""
        if hasattr(self, '_original_y_val') and len(y_true) != len(y_pred):
            # Use sequenced labels for consistent dimensions
            y_true = self._original_y_val
        return super().calculate_metrics(y_true, y_pred)
    
    def _save_model(self, model: Any, path: Path):
        """Save Keras model."""
        try:
            # Save in Keras format
            keras_path = path.parent / "model.h5"
            model.save(keras_path)
            
            # Also save in SavedModel format for deployment
            savedmodel_path = path.parent / "saved_model"
            model.save(savedmodel_path, save_format='tf')
            
            logger.debug("Model saved", keras_path=str(keras_path), 
                        savedmodel_path=str(savedmodel_path))
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise
    
    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM model."""
        # Create sequences for prediction
        if len(X) < self.sequence_length:
            logger.warning("Insufficient data for prediction",
                          data_length=len(X),
                          required_length=self.sequence_length)
            return np.array([1] * len(X))  # Default to neutral class
        
        X_seq, _ = self.create_sequences(X, np.zeros(len(X)))
        pred_proba = model.predict(X_seq, verbose=0)
        predictions = np.argmax(pred_proba, axis=1)
        
        # Pad predictions for the first sequence_length-1 samples
        padded_predictions = np.full(len(X), 1, dtype=np.int32)  # Default to neutral
        padded_predictions[self.sequence_length-1:] = predictions
        
        return padded_predictions


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train LSTM model for ATRX trading system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True,
                       help='Path to assembled dataset parquet file')
    parser.add_argument('--splits', required=True,
                       help='Directory containing time series splits')
    parser.add_argument('--features-config', required=True,
                       help='Path to features configuration YAML')
    parser.add_argument('--training-config', required=True,
                       help='Path to training configuration YAML')
    parser.add_argument('--outdir', required=True,
                       help='Output directory for models and predictions')
    
    # LSTM-specific arguments
    parser.add_argument('--seq-len', type=int, default=64,
                       help='Input sequence length')
    
    # Optional arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize trainer
        trainer = LSTMTrainer(
            dataset_path=args.dataset,
            splits_dir=args.splits,
            features_config_path=args.features_config,
            training_config_path=args.training_config,
            output_dir=args.outdir,
            sequence_length=args.seq_len
        )
        
        # Run training
        trainer.run_training()
        
        print(f"✅ LSTM training completed successfully!")
        print(f"📁 Models and predictions saved to: {args.outdir}")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
