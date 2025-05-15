import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
from ..utils.metrics import (
    calculate_metrics, plot_metrics_over_iterations,
    plot_threshold_metrics, save_metrics_comparison
)
from ..utils.feature_importance import analyze_feature_importance
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and evaluation for Dota 2 match prediction."""
    
    def __init__(self, save_path: Path, config: Dict[str, Any], categorical_features: List[str]):
        """
        Initialize the model trainer.
        
        Args:
            save_path: Path to save model artifacts and metrics
            config: Configuration dictionary containing model parameters
            categorical_features: List of categorical feature names
        """
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)  # Create save directory
        
        self.categorical_features = categorical_features
        # Get model configuration
        model_config = config.get('model', {})
        self.use_param_grid = model_config.get('use_param_grid', False)
        self.do_shap = model_config.get('do_shap', False)
        self.use_scaling = model_config.get('use_scaling', True)  # Add scaling config
        logger.info(f"Using parameter grid search: {self.use_param_grid}")
        logger.info(f"Using SHAP: {self.do_shap}")
        logger.info(f"Using feature scaling: {self.use_scaling}")
        
        # Initialize scaler
        self.scaler = StandardScaler() if self.use_scaling else None
        
        # Get parameter grids from config
        self.param_grids = model_config.get('param_grids', {})
        
        # Initialize base models with parameters from config
        base_params = model_config.get('base_params', {})
        self.base_models = {
            'LogisticRegression': LogisticRegression(**base_params.get('LogisticRegression', {})),
            'RandomForest': RandomForestClassifier(**base_params.get('RandomForest', {})),
            'CatBoost': CatBoostClassifier(**base_params.get('CatBoost', {}))
        }
        
        self.models = {}  # Will store either base models or grid search results
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.best_metric = 0.0
        
        # Store validation data for plotting
        self.X_val = None
        self.y_val = None
        
        # Store metrics history for CatBoost
        self.metrics_history = {}
        
    def _create_model(self, name: str) -> Any:
        """
        Create a model instance, either with or without parameter grid search.
        
        Args:
            name: Name of the model
            
        Returns:
            Model instance
        """
        if not self.use_param_grid:
            return self.base_models[name]
            
        logger.info(f"\nPerforming grid search for {name}...")
        grid_search = GridSearchCV(
            estimator=self.base_models[name],
            param_grid=self.param_grids[name],
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        return grid_search
        
    def _scale_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Scale non-categorical features using StandardScaler.
        
        Args:
            X: Input features DataFrame
            is_training: Whether this is training data (fit) or validation/test data (transform)
            
        Returns:
            Scaled features DataFrame
        """
        if not self.use_scaling:
            return X
            
        # Create a copy to avoid modifying the original
        X_scaled = X.copy()
        
        # Get non-categorical columns
        numeric_cols = [col for col in X.columns if col not in self.categorical_features]
        
        if is_training:
            # Fit and transform on training data
            X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            # Only transform on validation/test data
            X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
            
        return X_scaled

    def train_and_evaluate(self, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate multiple models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of model metrics
        """
        # Store validation data for plotting
        self.X_val = X_val
        self.y_val = y_val
        
        logger.info("\nTraining and evaluating models...")
        
        # Scale features if enabled
        if self.use_scaling:
            logger.info("Scaling features...")
            X_train = self._scale_features(X_train, is_training=True)
            X_val = self._scale_features(X_val, is_training=False)
        
        for name in self.base_models.keys():
            logger.info(f"\nTraining {name}...")
            
            # Create model instance
            model = self._create_model(name)
            
            # Train model
            if name == 'CatBoost':
                if self.use_param_grid:
                    model.fit(X_train, y_train)
                    best_params = model.best_params_
                    logger.info(f"Best parameters for {name}: {best_params}")
                    # Create new model with best parameters
                    model = CatBoostClassifier(**best_params, random_seed=42, verbose=50)
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
                    # Store metrics history
                    self.metrics_history[name] = model.get_evals_result()
                else:
                    # For non-grid search, ensure we have the right parameters
                    base_params = self.base_models[name].get_params()
                    # Remove parameters that we want to set explicitly
                    base_params.pop('random_seed', None)
                    base_params.pop('verbose', None)
                    if self.categorical_features:
                        cat_feature_indices = [X_train.columns.get_loc(col) for col in self.categorical_features]
                        logging.info(f"CatBoost categorical features names: {self.categorical_features}")
                        logger.info(f"CatBoost categorical features indices: {cat_feature_indices}")
                        base_params['cat_features'] = cat_feature_indices

                    model = CatBoostClassifier(
                        **base_params,
                        random_seed=42,
                        verbose=50
                    )
                    # Train with evaluation set
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50
                    )
                    # Store metrics history
                    self.metrics_history[name] = model.get_evals_result()
            else:
                if self.use_param_grid:
                    model.fit(X_train, y_train)
                    best_params = model.best_params_
                    logger.info(f"Best parameters for {name}: {best_params}")
                    # Create new model with best parameters
                    model = self.base_models[name].__class__(**best_params)
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
            
            # Store the trained model
            self.models[name] = model
            
            # Get predictions using the trained model
            y_pred = self.models[name].predict(X_val)
            y_proba = self.models[name].predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, y_proba, y_pred)
            
            # Log metrics
            logger.info(f"\n{name} metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")

            # Update best model
            if metrics['auc_roc'] > self.best_metric:
                self.best_metric = metrics['auc_roc']
                self.best_model = self.models[name]
                self.best_model_name = name
                logger.info(f"\nNew best model: {name} (AUC-ROC: {metrics['auc_roc']:.4f})")
            
            self.metrics[name] = metrics

            if self.do_shap:
            # Analyze feature importance
                analyze_feature_importance(
                    model=self.models[name],
                    X=X_val,
                    model_name=name,
                    save_dir=self.save_path,
                    top_k=10
                )
        
        # Save model comparison
        save_metrics_comparison(self.metrics, self.save_path)
        
        # Generate plots
        self._generate_plots()
        
        return self.metrics
    
    def _generate_plots(self) -> None:
        """Generate and save plots for all models."""
        plots_dir = self.save_path / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            # Create model-specific directory
            model_dir = plots_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Plot metrics over iterations
            plot_metrics_over_iterations(
                model_name, model, 
                self.metrics_history.get(model_name, {}),
                self.metrics[model_name],
                self.X_val, self.y_val,
                model_dir
            )
            
            # Plot threshold metrics
            plot_threshold_metrics(
                model_name, model,
                self.X_val, self.y_val,
                model_dir
            )
    
    def save_best_model(self, model_name: str) -> None:
        """
        Save all trained models and the best model.
        
        Args:
            model_name: Base name for saving models
        """
        if not self.metrics:
            raise ValueError("No models have been trained yet")
        
        # Create checkpoints directory in run directory
        checkpoints_dir = self.save_path / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all models
        for name, model in self.models.items():
            if name in self.metrics:  # Only save models that were trained
                # Save in checkpoints directory
                if name == 'CatBoost':
                    model_path = checkpoints_dir / f"{model_name}_{name.lower()}.cbm"
                    model.save_model(str(model_path), format="cbm")
                else:
                    model_path = checkpoints_dir / f"{model_name}_{name.lower()}.joblib"
                    joblib.dump(model, model_path)
                
                logger.info(f"Saved {name} model to {model_path}")
        
        # Save best model to production directory
        # models_dir = Path('/Users/ankamenskiy/SmartDota/models')
        # # models_dir.mkdir(parents=True, exist_ok=True)
        
        # if self.best_model_name == 'CatBoost':
        #     prod_model_path = models_dir / 'catboost' / f"{model_name}.cbm"
        #     prod_model_path.parent.mkdir(parents=True, exist_ok=True)
        #     self.best_model.save_model(str(prod_model_path), format="cbm")
        # else:
        #     prod_model_path = models_dir / 'sklearn' / f"{model_name}.joblib"
        #     prod_model_path.parent.mkdir(parents=True, exist_ok=True)
        #     joblib.dump(self.best_model, prod_model_path)
        
        # logger.info(f"Saved best model ({self.best_model_name}) to production at {prod_model_path}")
    
    def predict_match(self, match_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Make a prediction for a single match.
        
        Args:
            match_data: Dictionary containing match data with hero picks
            
        Returns:
            Dictionary with prediction results
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Extract hero picks
        radiant_picks = match_data['radiant_team']
        dire_picks = match_data['dire_team']
        
        # Create feature vector
        feature_vector = np.zeros(2 * 124, dtype=np.int8)  # 124 heroes per team
        
        # Set Radiant heroes
        for hero_id in radiant_picks:
            feature_vector[hero_id - 1] = 1
        
        # Set Dire heroes
        for hero_id in dire_picks:
            feature_vector[124 + hero_id - 1] = 1
        
        # Make prediction
        pred = self.best_model.predict([feature_vector])[0]
        probas = self.best_model.predict_proba([feature_vector])[0]
        
        return {
            'result': 'Radiant' if pred == 1 else 'Dire',
            'dire': f'{probas[0]:.2f}',
            'radiant': f'{probas[1]:.2f}'
        } 