import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss
)
import logging
import json

logger = logging.getLogger(__name__)

# Load hero mapping
def load_hero_mapping() -> Dict[int, str]:
    """Load hero ID to localized name mapping from heroes.json."""
    try:
        with open('data/heroes.json', 'r') as f:
            heroes = json.load(f)
            return {hero['id']: hero['localized_name'] for hero in heroes}
    except Exception as e:
        logger.warning(f"Failed to load hero mapping: {e}")
        return {}

# Load hero mapping at module level
HERO_MAPPING = load_hero_mapping()

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
        'log_loss': float(log_loss(y_true, y_pred_proba))
    }

def plot_metrics_over_iterations(trainer, save_path: Path) -> None:
    """
    Plot training and validation metrics over iterations for all models.
    
    Args:
        trainer: ModelTrainer instance with trained models and metrics history
        save_path: Path to save the plots
    """
    for model_name, model in trainer.models.items():
        plt.figure(figsize=(12, 8))
        
        if model_name == 'CatBoost':
            # Plot CatBoost metrics from history
            metrics = trainer.metrics_history[model_name]
            for metric_name, values in metrics.items():
                if 'train' in metric_name:
                    plt.plot(values, label=metric_name, alpha=0.7)
                if 'val' in metric_name:
                    plt.plot(values, label=metric_name, alpha=0.7)
            
            # Add final validation loss line
            if 'val_log_loss' in metrics:
                final_val_loss = metrics['val_log_loss'][-1]
                plt.axhline(y=final_val_loss, color='r', linestyle='--', 
                           label=f'Final Val Loss: {final_val_loss:.4f}')
            
            plt.title(f'{model_name} Metrics Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Log Loss')
            
        else:
            # For other models, plot validation metrics
            y_pred_proba = model.predict_proba(trainer.X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = calculate_metrics(trainer.y_val, y_pred_proba, y_pred)
            
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
        
        # Create model-specific directory
        model_dir = save_path / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save plot
        plt.savefig(model_dir / 'metrics_over_iterations.png')
        plt.close()

def plot_threshold_metrics(trainer, save_path: Path) -> None:
    """
    Plot metrics for different threshold values for all models.
    
    Args:
        trainer: ModelTrainer instance with trained models
        save_path: Path to save the plots
    """
    for model_name, model in trainer.models.items():
        plt.figure(figsize=(10, 6))
        
        # Get predictions
        y_pred_proba = model.predict_proba(trainer.X_val)[:, 1]
        
        # Calculate metrics for different thresholds
        thresholds = np.linspace(0, 1, 100)  # More granular thresholds
        metrics = {
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics['f1'].append(f1_score(trainer.y_val, y_pred))
            metrics['precision'].append(precision_score(trainer.y_val, y_pred))
            metrics['recall'].append(recall_score(trainer.y_val, y_pred))
        
        # Plot metrics
        plt.plot(thresholds, metrics['f1'], label='F1 Score', color='blue', linewidth=2)
        plt.plot(thresholds, metrics['precision'], label='Precision', color='red', linewidth=2)
        plt.plot(thresholds, metrics['recall'], label='Recall', color='green', linewidth=2)
        
        # Add vertical line at default threshold
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold (0.5)')
        
        # Find and mark optimal F1 threshold
        optimal_idx = np.argmax(metrics['f1'])
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='blue', linestyle=':', alpha=0.5,
                   label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')
        
        plt.title(f'{model_name} Metrics vs Classification Threshold')
        plt.xlabel('Classification Threshold')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create model-specific directory
        model_dir = save_path / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save plot
        plt.savefig(model_dir / 'threshold_metrics.png')
        plt.close()
        
        # Log optimal threshold and metrics
        logger.info(f"\n{model_name} optimal threshold analysis:")
        logger.info(f"Optimal F1 threshold: {optimal_threshold:.3f}")
        logger.info(f"Metrics at optimal threshold:")
        logger.info(f"F1 Score: {metrics['f1'][optimal_idx]:.3f}")
        logger.info(f"Precision: {metrics['precision'][optimal_idx]:.3f}")
        logger.info(f"Recall: {metrics['recall'][optimal_idx]:.3f}")
        
        # Log hero mapping if available
        if HERO_MAPPING:
            logger.info("\nHero ID to Name Mapping:")
            for hero_id, hero_name in sorted(HERO_MAPPING.items()):
                logger.info(f"ID {hero_id}: {hero_name}")

def save_metrics_comparison(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    """
    Save model comparison metrics to CSV and JSON files.
    
    Args:
        results: Dictionary of model metrics
        save_path: Path to save the comparison files
    """
    # Convert numpy values to Python native types
    results_to_save = {}
    for model_name, metrics in results.items():
        results_to_save[model_name] = {
            metric: float(value) if isinstance(value, np.ndarray) else value
            for metric, value in metrics.items()
        }
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results_to_save).T
    
    # Save as CSV
    csv_path = save_path / "model_comparison.csv"
    comparison_df.to_csv(csv_path)
    logger.info(f"Saved model comparison to {csv_path}")
    
    # Save as JSON
    json_path = save_path / "model_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f"Saved model comparison to {json_path}")
    
    # Log comparison
    logger.info("\nModel comparison:")
    logger.info("\n" + str(comparison_df)) 