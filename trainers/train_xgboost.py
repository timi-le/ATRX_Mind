#!/usr/bin/env python3
"""
ATRX XGBoost Training Script
===========================

Trains XGBoost models for financial time series prediction using walk-forward validation.
Implements gradient boosting with comprehensive hyperparameter optimization.

Following ATRX development standards for production-ready ML training.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from trainers.base_trainer import BaseTrainer

logger = structlog.get_logger(__name__)


class XGBoostTrainer(BaseTrainer):
    """XGBoost model trainer for financial time series."""
    
    def __init__(self, *args, **kwargs):
        """Initialize XGBoost trainer."""
        super().__init__(*args, model_type='xgboost', **kwargs)
        
        # Set random seed for reproducibility
        np.random.seed(self.global_config.get('seed', 42))
    
    def create_dmatrix(self, X: np.ndarray, y: np.ndarray = None) -> xgb.DMatrix:
        """
        Create XGBoost DMatrix for efficient training.
        
        Args:
            X: Feature array
            y: Label array (optional)
            
        Returns:
            XGBoost DMatrix
        """
        if y is not None:
            return xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        else:
            return xgb.DMatrix(X, feature_names=self.feature_names)
    
    def get_xgb_params(self) -> dict:
        """Get XGBoost training parameters."""
        config = self.model_config
        
        params = {
            # Core parameters
            'objective': config.get('objective', 'multi:softprob'),
            'num_class': config.get('num_class', 3),
            'eval_metric': config.get('eval_metric', ['mlogloss', 'merror']),
            
            # Tree parameters
            'max_depth': config.get('max_depth', 6),
            'min_child_weight': config.get('min_child_weight', 1),
            'gamma': config.get('gamma', 0.1),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'colsample_bylevel': config.get('colsample_bylevel', 0.8),
            
            # Regularization
            'reg_alpha': config.get('reg_alpha', 0.1),
            'reg_lambda': config.get('reg_lambda', 1.0),
            
            # Training parameters
            'learning_rate': config.get('learning_rate', 0.1),
            'verbosity': config.get('verbosity', 1),
            
            # Advanced
            'scale_pos_weight': config.get('scale_pos_weight', 1),
            'max_delta_step': config.get('max_delta_step', 0),
            'tree_method': config.get('tree_method', 'auto'),
            'predictor': config.get('predictor', 'auto'),
            
            # Random seed
            'random_state': self.global_config.get('seed', 42),
            'seed': self.global_config.get('seed', 42)
        }
        
        # Handle GPU training if available
        if config.get('tree_method') == 'gpu_hist':
            try:
                # Check if GPU is available
                import cupy
                params['tree_method'] = 'gpu_hist'
                params['predictor'] = 'gpu_predictor'
                logger.info("GPU training enabled")
            except ImportError:
                logger.warning("cupy not available, falling back to CPU")
                params['tree_method'] = 'hist'
        
        return params
    
    def train_fold(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Train XGBoost model for one fold."""
        
        logger.info("Preparing XGBoost data",
                   train_samples=len(X_train),
                   val_samples=len(X_val),
                   features=X_train.shape[1])
        
        # Create DMatrix objects
        dtrain = self.create_dmatrix(X_train, y_train)
        dval = self.create_dmatrix(X_val, y_val)
        
        # Set up evaluation list
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        # Get training parameters
        params = self.get_xgb_params()
        num_rounds = self.model_config.get('n_estimators', 1000)
        early_stopping_rounds = self.model_config.get('early_stopping_rounds', 50)
        
        # Progress tracking
        evals_result = {}
        
        logger.info("Starting XGBoost training",
                   num_rounds=num_rounds,
                   early_stopping_rounds=early_stopping_rounds,
                   params=params)
        
        # Train model
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=evallist,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50  # Print every 50 rounds
        )
        
        # Make predictions
        val_pred_proba = model.predict(dval)
        val_predictions = np.argmax(val_pred_proba, axis=1)
        
        # Log feature importance
        importance = model.get_score(importance_type='weight')
        logger.info("Training completed",
                   best_iteration=model.best_iteration,
                   best_score=model.best_score,
                   num_features_used=len(importance))
        
        # Save feature importance for this fold
        fold_id = len(self.fold_results)
        self._save_feature_importance(model, fold_id)
        
        return model, val_predictions
    
    def _save_feature_importance(self, model: xgb.Booster, fold_id: int):
        """Save feature importance for analysis."""
        fold_dir = self.output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        # Get different importance types
        importance_types = ['weight', 'gain', 'cover']
        importance_data = {}
        
        for imp_type in importance_types:
            try:
                importance = model.get_score(importance_type=imp_type)
                importance_data[imp_type] = importance
            except Exception as e:
                logger.warning(f"Could not get {imp_type} importance", error=str(e))
        
        # Save as JSON
        import json
        json_path = fold_dir / "feature_importance.json"
        with open(json_path, 'w') as f:
            json.dump(importance_data, f, indent=2)
        
        # Save as DataFrame for easy analysis
        if importance_data:
            csv_path = fold_dir / "feature_importance.csv"
            importance_df = pd.DataFrame(importance_data).fillna(0)
            importance_df.to_csv(csv_path)
    
    def _save_model(self, model: Any, path: Path):
        """Save XGBoost model."""
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save XGBoost native format
            xgb_path = path.parent / "model.xgb"
            model.save_model(str(xgb_path))
            
            # Save as pickle for Python compatibility
            pickle_path = path.parent / "model.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save model dump for analysis (use get_dump instead of get_dump_text)
            try:
                dump_path = path.parent / "model_dump.txt"
                dump_list = model.get_dump()
                with open(dump_path, 'w') as f:
                    for i, tree in enumerate(dump_list):
                        f.write(f"Tree {i}:\n{tree}\n\n")
            except Exception as e:
                logger.warning("Could not save model dump", error=str(e))
            
            logger.debug("Model saved", xgb_path=str(xgb_path), 
                        pickle_path=str(pickle_path))
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise
    
    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost model."""
        dtest = self.create_dmatrix(X)
        pred_proba = model.predict(dtest)
        
        # Handle both probability and class prediction formats
        if pred_proba.ndim == 2:
            predictions = np.argmax(pred_proba, axis=1)
        else:
            predictions = pred_proba.astype(int)
        
        return predictions
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> dict:
        """Calculate comprehensive evaluation metrics for XGBoost."""
        # Get base metrics
        metrics = super().calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Add XGBoost-specific metrics
        try:
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Add per-class metrics - check if classes exist in data
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            for label in unique_labels:
                label_str = str(int(label))
                if label_str in report:
                    metrics[f'precision_class_{label_str}'] = report[label_str]['precision']
                    metrics[f'recall_class_{label_str}'] = report[label_str]['recall']
                    metrics[f'f1_class_{label_str}'] = report[label_str]['f1-score']
            
            # Add macro averages
            if 'macro avg' in report:
                metrics['precision_macro'] = report['macro avg']['precision']
                metrics['recall_macro'] = report['macro avg']['recall']
                metrics['f1_macro'] = report['macro avg']['f1-score']
        
        except Exception as e:
            logger.warning("Could not calculate detailed metrics", error=str(e))
        
        return metrics
    
    def save_summary(self):
        """Save training summary with XGBoost-specific information."""
        # Call parent method
        super().save_summary()
        
        # Add XGBoost-specific summary
        try:
            # Aggregate feature importance across folds
            all_importance = {}
            importance_types = ['weight', 'gain', 'cover']
            
            for imp_type in importance_types:
                all_importance[imp_type] = {}
            
            # Collect importance from all folds
            for fold_id in range(len(self.fold_results)):
                fold_dir = self.output_dir / f"fold_{fold_id}"
                importance_file = fold_dir / "feature_importance.json"
                
                if importance_file.exists():
                    try:
                        import json
                        with open(importance_file, 'r') as f:
                            fold_importance = json.load(f)
                        
                        for imp_type in importance_types:
                            if imp_type in fold_importance:
                                for feature, value in fold_importance[imp_type].items():
                                    if feature not in all_importance[imp_type]:
                                        all_importance[imp_type][feature] = []
                                    all_importance[imp_type][feature].append(value)
                    except Exception as e:
                        logger.warning(f"Could not load importance for fold {fold_id}", error=str(e))
            
            # Calculate average importance
            avg_importance = {}
            for imp_type in importance_types:
                avg_importance[imp_type] = {}
                for feature, values in all_importance[imp_type].items():
                    avg_importance[imp_type][feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'folds': len(values)
                    }
            
            # Save aggregated feature importance
            import json
            with open(self.output_dir / "feature_importance_summary.json", 'w') as f:
                json.dump(avg_importance, f, indent=2, default=str)
            
            logger.info("XGBoost summary saved with feature importance analysis")
        
        except Exception as e:
            logger.warning("Could not create XGBoost-specific summary", error=str(e))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost model for ATRX trading system',
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
    
    # Optional arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU training (requires cupy)')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize trainer
        trainer = XGBoostTrainer(
            dataset_path=args.dataset,
            splits_dir=args.splits,
            features_config_path=args.features_config,
            training_config_path=args.training_config,
            output_dir=args.outdir
        )
        
        # Override GPU setting if requested
        if args.gpu:
            trainer.model_config['tree_method'] = 'gpu_hist'
        
        # Run training
        trainer.run_training()
        
        print(f"✅ XGBoost training completed successfully!")
        print(f"📁 Models and predictions saved to: {args.outdir}")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
