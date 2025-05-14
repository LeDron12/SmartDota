import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Tuple
from catboost import CatBoostClassifier

logging.getLogger('shap').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def analyze_feature_importance(model: Any, X: pd.DataFrame, model_name: str, save_dir: Path, top_k: int = 10) -> Dict[str, Any]:
    """
    Analyze and plot feature importance for a model.
    
    Args:
        model: Trained model instance
        X: Feature DataFrame
        model_name: Name of the model
        save_dir: Directory to save plots and results
        top_k: Number of top features to show
        
    Returns:
        Dictionary with feature importance statistics
    """
    logger.info(f"\nAnalyzing feature importance for {model_name}")
    
    # Create model-specific directory
    model_dir = save_dir / 'plots' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, CatBoostClassifier):
        importance_dict = analyze_catboost_importance(model, X, model_name, model_dir, top_k)
    else:
        importance_dict = analyze_shap_importance(model, X, model_name, model_dir, top_k)
    
    # Save importance statistics
    save_importance_stats(importance_dict, model_dir, model_name)
    
    return importance_dict

def analyze_catboost_importance(model: CatBoostClassifier, X: pd.DataFrame, model_name: str, 
                              save_dir: Path, top_k: int) -> Dict[str, Any]:
    """Analyze feature importance for CatBoost model."""
    # Get feature importance
    importance = model.get_feature_importance()
    feature_names = X.columns
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot importance
    plt.figure(figsize=(12, 8))
    
    # Plot top K features
    plt.subplot(2, 1, 1)
    top_features = importance_df.head(top_k)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title(f'{model_name} - Top {top_k} Important Features')
    plt.xlabel('Importance')
    
    # Plot bottom K features
    plt.subplot(2, 1, 2)
    bottom_features = importance_df.tail(top_k)
    plt.barh(bottom_features['feature'], bottom_features['importance'])
    plt.title(f'{model_name} - Bottom {top_k} Important Features')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png')
    plt.close()
    
    return {
        'top_features': top_features.to_dict('records'),
        'bottom_features': bottom_features.to_dict('records'),
        'all_features': importance_df.to_dict('records')
    }

def analyze_shap_importance(model: Any, X: pd.DataFrame, model_name: str, 
                          save_dir: Path, top_k: int) -> Dict[str, Any]:
    """Analyze feature importance using SHAP values."""
    # Calculate SHAP values using appropriate explainer
    if hasattr(model, 'predict_proba'):
        # For sklearn models, use KernelExplainer with a subset of data as background
        # Instead of k-means, use a stratified sample to ensure diverse background
        n_background = min(10, len(X))  # Use at most 100 samples for background
        if len(X) > n_background:
            background = X.sample(n=n_background, random_state=42)
        else:
            background = X
            
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)
        # For binary classification, use values for positive class
        shap_values = shap_values[:, :,1]
    else:
        # For other models, use TreeExplainer if possible
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_values = shap_values[:, :,1]
        except:
            # Fallback to KernelExplainer with stratified background
            n_background = min(10, len(X))
            if len(X) > n_background:
                background = X.sample(n=n_background, random_state=42)
            else:
                background = X
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X)
    
    # Ensure shap_values is 2D array
    if len(shap_values.shape) > 2:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)
    
    # Get mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Ensure feature names match SHAP values
    feature_names = X.columns[:mean_shap.shape[0]]
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    # Plot importance
    plt.figure(figsize=(12, 8))
    
    # Plot top K features
    plt.subplot(2, 1, 1)
    top_features = importance_df.head(top_k)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title(f'{model_name} - Top {top_k} Important Features (SHAP)')
    plt.xlabel('Mean |SHAP value|')
    
    # Plot bottom K features
    plt.subplot(2, 1, 2)
    bottom_features = importance_df.tail(top_k)
    plt.barh(bottom_features['feature'], bottom_features['importance'])
    plt.title(f'{model_name} - Bottom {top_k} Important Features (SHAP)')
    plt.xlabel('Mean |SHAP value|')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png')
    plt.close()
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 8))
    # Use only the features that have SHAP values
    X_shap = X[feature_names]
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title(f'{model_name} - SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(save_dir / 'shap_summary.png')
    plt.close()
    
    return {
        'top_features': top_features.to_dict('records'),
        'bottom_features': bottom_features.to_dict('records'),
        'all_features': importance_df.to_dict('records')
    }

def save_importance_stats(importance_dict: Dict[str, Any], save_dir: Path, model_name: str) -> None:
    """Save feature importance statistics to JSON file."""
    # Save importance statistics
    importance_file = save_dir / 'feature_importance.json'
    with open(importance_file, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    logger.info(f"Saved feature importance statistics for {model_name}") 