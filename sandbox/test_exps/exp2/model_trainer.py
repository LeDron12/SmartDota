import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from pathlib import Path
from metrics import calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, save_path: str = "models"):
        """
        Initialize the model trainer.
        
        Args:
            save_path: Path to save model results
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {
            'CatBoost': CatBoostClassifier(
                iterations=1000,
                learning_rate=0.07,
                depth=5,
                loss_function='Logloss',
                verbose=50,
                eval_metric='Logloss',
                train_dir=str(self.save_path / 'catboost_tmp'),
                allow_writing_files=False
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        self.results = {}
        self.metrics_history = {}
        self.best_metrics = {}
        self.X_val = None
        self.y_val = None
    
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of model results
        """
        logger.info("Starting model training and evaluation...")
        
        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")
            
            # Initialize metrics history for this model
            self.metrics_history[name] = {
                'train_log_loss': [],
                'val_log_loss': []
            }
            
            # Train model
            if name == 'CatBoost':
                # For CatBoost, use eval_set for validation metrics
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=200
                )
                
                # Get metrics history
                eval_results = model.get_evals_result()
                if 'learn' in eval_results:
                    self.metrics_history[name]['train_log_loss'] = eval_results['learn']['Logloss']
                if 'validation' in eval_results:
                    self.metrics_history[name]['val_log_loss'] = eval_results['validation']['Logloss']
            else:
                model.fit(X_train, y_train)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred_proba, y_pred)
            self.results[name] = metrics
            self.best_metrics[name] = metrics
            
            # Log metrics
            logger.info(f"{name} metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
            # Save model
            model_path = self.save_path / name
            model_path.mkdir(exist_ok=True)
            if name == 'CatBoost':
                model.save_model(str(model_path / 'model.cbm'))
            else:
                import joblib
                joblib.dump(model, model_path / 'model.joblib')
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing model based on AUC-ROC score.
        
        Returns:
            Tuple of (model_name, metrics)
        """
        best_model = max(
            self.results.items(),
            key=lambda x: x[1]['auc_roc']
        )
        return best_model 