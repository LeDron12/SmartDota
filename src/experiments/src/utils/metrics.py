import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, log_loss, accuracy_score, brier_score_loss
)

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        'auc_roc': float(roc_auc_score(y_true, y_pred_proba)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'log_loss': float(log_loss(y_true, y_pred_proba)),
        'brier_score': float(brier_score_loss(y_true, y_pred_proba))
    }

def plot_metrics_over_iterations(model_name: str, model: Any, metrics_history: Dict[str, List[float]], 
                               metrics: Dict[str, float], X_val: pd.DataFrame, y_val: pd.Series, 
                               save_dir: Path) -> None:
    """
    Plot training and validation metrics over iterations.
    
    Args:
        model_name: Name of the model
        model: Trained model instance
        metrics_history: Dictionary of metrics history for CatBoost
        metrics: Dictionary of model metrics
        X_val: Validation features
        y_val: Validation labels
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    if model_name == 'CatBoost' and metrics_history:
        # Plot CatBoost metrics from history
        for metric_name, values in metrics_history.items():
            if isinstance(values, dict):
                # Handle nested dictionary structure from CatBoost
                for subset_name, subset_values in values.items():
                    if isinstance(subset_values, list):
                        plt.plot(subset_values, label=f'{metric_name}_{subset_name}', alpha=0.7)
                    elif isinstance(subset_values, np.ndarray):
                        plt.plot(subset_values.tolist(), label=f'{metric_name}_{subset_name}', alpha=0.7)
        
        # Add final validation loss line if available
        if 'validation' in metrics_history.get('Logloss', {}):
            final_val_loss = metrics_history['Logloss']['validation'][-1]
            plt.axhline(y=final_val_loss, color='r', linestyle='--', 
                       label=f'Final Val Loss: {final_val_loss:.4f}')
        
        plt.title(f'{model_name} Metrics Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        
        # Log available metrics for debugging
        # logger.info(f"\nAvailable CatBoost metrics:")
        # for metric_name, values in metrics_history.items():
        #     logger.info(f"{metric_name}: {type(values)}")
        #     if isinstance(values, dict):
        #         for subset_name, subset_values in values.items():
        #             logger.info(f"  {subset_name}: {type(subset_values)}, length: {len(subset_values) if hasattr(subset_values, '__len__') else 'N/A'}")
        
    else:
        # For other models, plot validation metrics
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Create bar plot
        plt.bar(metrics.keys(), metrics.values())
        plt.title(f'{model_name} Validation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for i, (metric_name, value) in enumerate(metrics.items()):
            plt.text(i, value + 0.02, f'{value:.3f}', ha='center')
    
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_dir / 'metrics_over_iterations.png')
    plt.close()

def plot_threshold_metrics(model_name: str, model: Any, X_val: pd.DataFrame, y_val: pd.Series, 
                          save_dir: Path) -> None:
    """
    Plot metrics for different threshold values.
    
    Args:
        model_name: Name of the model
        model: Trained model instance
        X_val: Validation features
        y_val: Validation labels
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics for different thresholds
    thresholds = np.linspace(0, 1, 100)
    metrics = {
        'f1': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'brier_score': []
    }
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['brier_score'].append(brier_score_loss(y_val, y_pred_proba))
    
    # Plot metrics
    plt.plot(thresholds, metrics['f1'], label='F1 Score', color='blue', linewidth=2)
    plt.plot(thresholds, metrics['precision'], label='Precision', color='red', linewidth=2)
    plt.plot(thresholds, metrics['recall'], label='Recall', color='green', linewidth=2)
    plt.plot(thresholds, metrics['accuracy'], label='Accuracy', color='purple', linewidth=2)
    plt.plot(thresholds, metrics['brier_score'], label='Brier Score', color='orange', linewidth=2)
    
    # Add vertical line at default threshold
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold (0.5)')

    optimal_idx_f1, optimal_threshold_f1 = _plot_optimal_threshold(metrics, thresholds, 'f1', 'blue')
    optimal_idx_accuracy, optimal_threshold_accuracy = _plot_optimal_threshold(metrics, thresholds, 'accuracy', 'green')
    # _, _ = _plot_optimal_threshold(metrics, thresholds, 'brier_score', 'orange')
    
    plt.title(f'{model_name} Metrics vs Classification Threshold')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_dir / 'threshold_metrics.png')
    plt.close()
    
    # Log optimal threshold and metrics
    logger.info(f"\n{model_name} optimal threshold analysis:")
    logger.info(f"Optimal F1 threshold: {optimal_threshold_f1:.3f}")
    logger.info(f"Metrics at optimal F1 threshold:")
    logger.info(f"F1 Score: {metrics['f1'][optimal_idx_f1]:.3f}")
    logger.info(f"Precision: {metrics['precision'][optimal_idx_f1]:.3f}")
    logger.info(f"Recall: {metrics['recall'][optimal_idx_f1]:.3f}")
    logger.info(f"Accuracy: {metrics['accuracy'][optimal_idx_f1]:.3f}")
    logger.info(f"Brier Score: {metrics['brier_score'][optimal_idx_f1]:.3f}")
    logger.info(f"\nOptimal Accuracy threshold: {optimal_threshold_accuracy:.3f}")
    logger.info(f"Metrics at optimal Accuracy threshold:")
    logger.info(f"F1 Score: {metrics['f1'][optimal_idx_accuracy]:.3f}")
    logger.info(f"Precision: {metrics['precision'][optimal_idx_accuracy]:.3f}")
    logger.info(f"Recall: {metrics['recall'][optimal_idx_accuracy]:.3f}")
    logger.info(f"Accuracy: {metrics['accuracy'][optimal_idx_accuracy]:.3f}")
    logger.info(f"Brier Score: {metrics['brier_score'][optimal_idx_accuracy]:.3f}")

def _plot_optimal_threshold(metrics, thresholds, metric_key: str, color: str) -> Tuple[int, float]:
    """Plot optimal threshold line and return its index and value."""
    optimal_idx = np.argmax(metrics[metric_key])
    optimal_threshold = thresholds[optimal_idx]
    plt.axvline(x=optimal_threshold, color=color, linestyle=':', alpha=0.5,
                label=f'Optimal {metric_key.upper()} Threshold ({optimal_threshold:.2f})')
    return optimal_idx, optimal_threshold

def save_metrics_comparison(metrics: Dict[str, Dict[str, float]], save_path: Path) -> None:
    """
    Save model comparison metrics to CSV and JSON files.
    
    Args:
        metrics: Dictionary of model metrics
        save_path: Path to save the comparison files
    """
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).T
    
    # Save as CSV
    csv_path = save_path / 'model_comparison.csv'
    metrics_df.to_csv(csv_path)
    logger.info(f"Saved model comparison to {csv_path}")
    
    # Save as JSON
    json_path = save_path / 'model_comparison.json'
    metrics_df.to_json(json_path)
    logger.info(f"Saved model comparison to {json_path}")
    
    # Log comparison
    logger.info("\nModel comparison:")
    logger.info(f"\n{metrics_df}") 