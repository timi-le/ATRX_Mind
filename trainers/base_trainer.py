"""
ATRX Base Training Module
========================

Shared utilities and base classes for all model training scripts.
Following ATRX development standards for consistent ML workflows.
"""

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report
)
import structlog

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = structlog.get_logger(__name__)


class BaseTrainer(ABC):
    """Base class for all ATRX model trainers."""
    
    def __init__(self, 
                 dataset_path: str,
                 splits_dir: str,
                 features_config_path: str,
                 training_config_path: str,
                 output_dir: str,
                 model_type: str):
        """
        Initialize base trainer.
        
        Args:
            dataset_path: Path to assembled dataset parquet file
            splits_dir: Directory containing time series splits
            features_config_path: Path to features configuration
            training_config_path: Path to training configuration  
            output_dir: Directory to save models and predictions
            model_type: Type of model ('lstm', 'cnn', 'xgboost')
        """
        self.dataset_path = Path(dataset_path)
        self.splits_dir = Path(splits_dir)
        self.features_config_path = Path(features_config_path)
        self.training_config_path = Path(training_config_path)
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.features_config = self._load_yaml(features_config_path)
        self.training_config = self._load_yaml(training_config_path)
        
        # Model-specific config
        self.model_config = self.training_config.get(model_type, {})
        self.global_config = self.training_config.get('global', {})
        
        # Initialize tracking
        self.fold_results = []
        self.oof_predictions = None
        self.feature_names = None
        self.scalers = {}
        
        logger.info("Base trainer initialized",
                   model_type=model_type,
                   dataset=str(dataset_path),
                   splits_dir=str(splits_dir),
                   output_dir=str(output_dir))
    
    def _load_yaml(self, path: Union[str, Path]) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load config", path=str(path), error=str(e))
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate dataset."""
        logger.info("Loading dataset", path=str(self.dataset_path))
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        df = pd.read_parquet(self.dataset_path)
        
        # Validate required columns
        required_cols = ['timestamp', 'symbol', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get feature columns
        self.feature_names = [col for col in df.columns 
                             if col not in ['timestamp', 'symbol', 'label']]
        
        logger.info("Dataset loaded successfully",
                   rows=len(df),
                   features=len(self.feature_names),
                   symbols=df['symbol'].nunique(),
                   date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def load_splits(self) -> List[Dict]:
        """Load time series splits."""
        # Find splits file
        splits_files = list(self.splits_dir.glob("*_splits.json"))
        if not splits_files:
            raise FileNotFoundError(f"No splits file found in {self.splits_dir}")
        
        splits_file = splits_files[0]  # Take the first one
        logger.info("Loading splits", file=str(splits_file))
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        logger.info("Splits loaded successfully", num_splits=len(splits))
        return splits
    
    def prepare_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Prepare features for training with model-specific NaN handling."""
        # Select feature columns
        X_features = X[self.feature_names].copy()
        
        # Model-specific NaN handling
        model_type = self.__class__.__name__.lower()
        
        if X_features.isnull().any().any():
            nan_count = X_features.isnull().sum().sum()
            logger.info("Found NaN values in features", 
                       nan_count=nan_count,
                       model_type=model_type)
            
            # XGBoost can handle NaN values natively, others need filling
            if 'xgboost' in model_type:
                logger.info("XGBoost detected: preserving NaN values for native handling")
                # Keep NaN as-is for XGBoost native handling
                pass
            else:
                logger.info("Neural network detected: forward/backward filling NaN values")
                # For LSTM/CNN, use forward/backward fill
                for col in X_features.columns:
                    if X_features[col].isnull().any():
                        X_features[col] = X_features[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            logger.debug("No NaN values found in features")
        
        # Scale features if configured
        if self.training_config.get('feature_engineering', {}).get('scale_features', True):
            scaler_type = self.training_config.get('feature_engineering', {}).get('scaler_type', 'standard')
            
            if fit_scaler:
                # Fit new scaler
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                elif scaler_type == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {scaler_type}")
                
                X_scaled = scaler.fit_transform(X_features)
                self.scalers['feature_scaler'] = scaler
                
                logger.debug("Fitted feature scaler", scaler_type=scaler_type)
            else:
                # Use existing scaler
                if 'feature_scaler' not in self.scalers:
                    raise ValueError("No fitted scaler available")
                X_scaled = self.scalers['feature_scaler'].transform(X_features)
        else:
            X_scaled = X_features.values
        
        return X_scaled
    
    def prepare_labels(self, y: pd.Series) -> np.ndarray:
        """Prepare labels for training."""
        # Encode labels: {-1, 0, 1} -> {0, 1, 2}
        if self.training_config.get('feature_engineering', {}).get('encode_labels', True):
            label_mapping = {-1: 0, 0: 1, 1: 2}
            y_encoded = y.map(label_mapping)
            
            # Check for unmapped values
            if y_encoded.isna().any():
                logger.warning("Found unmapped label values", 
                              unique_labels=y.unique().tolist())
                y_encoded = y_encoded.fillna(1)  # Default to neutral
        else:
            y_encoded = y
        
        return y_encoded.values
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                      multi_class='ovr', average='macro')
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except ValueError as e:
                logger.warning("Could not calculate probability metrics", error=str(e))
                metrics['roc_auc_ovr'] = 0.0
                metrics['log_loss'] = float('inf')
        
        # Custom financial metrics
        metrics['directional_accuracy'] = self._calculate_directional_accuracy(y_true, y_pred)
        
        return metrics
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (custom FX metric)."""
        # Focus on directional predictions (ignore neutral class 1)
        directional_mask = (y_true != 1) & (y_pred != 1)
        
        if directional_mask.sum() == 0:
            return 0.0
        
        y_true_dir = y_true[directional_mask]
        y_pred_dir = y_pred[directional_mask]
        
        return accuracy_score(y_true_dir, y_pred_dir)
    
    def save_fold_results(self, fold_id: int, model: Any, 
                         train_metrics: Dict, val_metrics: Dict,
                         val_predictions: np.ndarray, val_indices: List[int]):
        """Save results for a single fold."""
        fold_dir = self.output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = fold_dir / f"model.pkl"
        self._save_model(model, model_path)
        
        # Save metrics
        metrics = {
            'fold_id': fold_id,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'val_samples': len(val_indices)
        }
        
        with open(fold_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'index': val_indices,
            'prediction': val_predictions
        })
        pred_df.to_parquet(fold_dir / "predictions.parquet", index=False)
        
        self.fold_results.append(metrics)
        
        logger.info("Fold results saved",
                   fold_id=fold_id,
                   val_accuracy=val_metrics.get('accuracy', 0),
                   output_dir=str(fold_dir))
    
    def save_oof_predictions(self, dataset_length: int):
        """Compile and save out-of-fold predictions."""
        if not self.fold_results:
            logger.warning("No fold results to compile")
            return
        
        # Initialize OOF array
        oof_predictions = np.full(dataset_length, -999, dtype=np.float32)
        
        # Collect predictions from all folds
        for fold_id in range(len(self.fold_results)):
            fold_dir = self.output_dir / f"fold_{fold_id}"
            pred_file = fold_dir / "predictions.parquet"
            
            if pred_file.exists():
                pred_df = pd.read_parquet(pred_file)
                oof_predictions[pred_df['index']] = pred_df['prediction']
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            'index': range(dataset_length),
            'oof_prediction': oof_predictions
        })
        
        oof_path = self.output_dir / "oof_predictions.parquet"
        oof_df.to_parquet(oof_path, index=False)
        
        # Calculate coverage
        valid_predictions = (oof_predictions != -999).sum()
        coverage = valid_predictions / dataset_length
        
        logger.info("OOF predictions saved",
                   path=str(oof_path),
                   coverage=f"{coverage:.1%}",
                   valid_predictions=valid_predictions,
                   total_samples=dataset_length)
        
        self.oof_predictions = oof_predictions
    
    def save_summary(self):
        """Save training summary and aggregated metrics."""
        if not self.fold_results:
            logger.warning("No results to summarize")
            return
        
        # Aggregate metrics across folds
        train_metrics_avg = {}
        val_metrics_avg = {}
        
        # Get all unique metric names across all folds
        all_train_metrics = set()
        all_val_metrics = set()
        
        for fold in self.fold_results:
            all_train_metrics.update(fold['train_metrics'].keys())
            all_val_metrics.update(fold['val_metrics'].keys())
        
        # Aggregate train metrics
        for metric in all_train_metrics:
            train_values = []
            for fold in self.fold_results:
                if metric in fold['train_metrics']:
                    train_values.append(fold['train_metrics'][metric])
            
            if train_values:  # Only aggregate if we have values
                train_metrics_avg[metric] = {
                    'mean': np.mean(train_values),
                    'std': np.std(train_values),
                    'values': train_values,
                    'count': len(train_values)
                }
        
        # Aggregate validation metrics
        for metric in all_val_metrics:
            val_values = []
            for fold in self.fold_results:
                if metric in fold['val_metrics']:
                    val_values.append(fold['val_metrics'][metric])
            
            if val_values:  # Only aggregate if we have values
                val_metrics_avg[metric] = {
                    'mean': np.mean(val_values),
                    'std': np.std(val_values),
                    'values': val_values,
                    'count': len(val_values)
                }
        
        summary = {
            'model_type': self.model_type,
            'num_folds': len(self.fold_results),
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'training_config': self.model_config,
            'train_metrics': train_metrics_avg,
            'val_metrics': val_metrics_avg,
            'fold_results': self.fold_results
        }
        
        # Save summary
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save feature names
        if self.feature_names:
            with open(self.output_dir / "feature_names.json", 'w') as f:
                json.dump(self.feature_names, f, indent=2)
        
        # Save scalers
        if self.scalers:
            with open(self.output_dir / "scalers.pkl", 'wb') as f:
                pickle.dump(self.scalers, f)
        
        logger.info("Training summary saved",
                   val_accuracy_mean=val_metrics_avg.get('accuracy', {}).get('mean', 0),
                   val_accuracy_std=val_metrics_avg.get('accuracy', {}).get('std', 0),
                   num_folds=len(self.fold_results))
    
    @abstractmethod
    def _save_model(self, model: Any, path: Path):
        """Save model to disk. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train_fold(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Train model for one fold. Must be implemented by subclasses."""
        pass
    
    def run_training(self):
        """Main training loop."""
        logger.info("Starting training", model_type=self.model_type)
        
        # Load data and splits
        df = self.load_data()
        splits = self.load_splits()
        
        # Process each fold
        for fold_id, split in enumerate(splits):
            logger.info(f"Training fold {fold_id + 1}/{len(splits)}")
            
            # Get split information for the first symbol (assuming single symbol)
            symbol = list(split['splits'].keys())[0]
            split_info = split['splits'][symbol]
            
            train_indices = split_info['train_indices']
            val_indices = split_info['val_indices']
            
            # Prepare training data
            X_train = self.prepare_features(df.iloc[train_indices], fit_scaler=True)
            y_train = self.prepare_labels(df.iloc[train_indices]['label'])
            
            # Prepare validation data
            X_val = self.prepare_features(df.iloc[val_indices], fit_scaler=False)
            y_val = self.prepare_labels(df.iloc[val_indices]['label'])
            
            # Train fold
            model, val_predictions = self.train_fold(X_train, y_train, X_val, y_val)
            
            # Calculate metrics
            train_pred = self._predict(model, X_train)
            train_metrics = self.calculate_metrics(y_train, train_pred)
            val_metrics = self.calculate_metrics(y_val, val_predictions)
            
            # Save fold results
            self.save_fold_results(fold_id, model, train_metrics, val_metrics,
                                 val_predictions, val_indices)
        
        # Save aggregated results
        self.save_oof_predictions(len(df))
        self.save_summary()
        
        logger.info("Training completed successfully",
                   model_type=self.model_type,
                   num_folds=len(splits),
                   output_dir=str(self.output_dir))
    
    @abstractmethod
    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with model. Must be implemented by subclasses."""
        pass