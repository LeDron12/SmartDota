import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from src.experiments.src.dl.data.dataset import get_time_window_metrics

@dataclass
class Metrics:
    loss: float
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    window_metrics: List[Tuple[float, float]]  # List of (accuracy, auc) for each time window
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'auc': self.auc,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'window_metrics': [
                {'accuracy': acc, 'auc': auc}
                for acc, auc in self.window_metrics
            ]
        }

def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    loss: float,
    window_size: int = 10
) -> Metrics:
    """
    Calculate various metrics for model evaluation
    
    Args:
        predictions: Model predictions (batch_size, seq_len)
        labels: True labels (batch_size, seq_len)
        masks: Valid timestep masks (batch_size, seq_len)
        loss: Loss value
        window_size: Size of time windows for windowed metrics
    
    Returns:
        Metrics object containing all calculated metrics
    """
    # Move tensors to CPU and convert to numpy
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    
    # Filter out padded timesteps
    valid_idx = masks.astype(bool)
    valid_preds = predictions[valid_idx]
    valid_labels = labels[valid_idx]
    
    # Calculate overall metrics
    try:
        auc = roc_auc_score(valid_labels, valid_preds)
    except ValueError:
        auc = 0.5  # Default to random chance if only one class
    
    binary_preds = (valid_preds > 0.5).astype(int)
    accuracy = accuracy_score(valid_labels, binary_preds)
    precision = precision_score(valid_labels, binary_preds, zero_division=0)
    recall = recall_score(valid_labels, binary_preds, zero_division=0)
    f1 = f1_score(valid_labels, binary_preds, zero_division=0)
    
    # Calculate windowed metrics
    window_metrics = get_time_window_metrics(
        torch.from_numpy(predictions),
        torch.from_numpy(labels),
        torch.from_numpy(masks),
        window_size
    )
    
    return Metrics(
        loss=loss,
        accuracy=accuracy,
        auc=auc,
        precision=precision,
        recall=recall,
        f1=f1,
        window_metrics=window_metrics
    )

class MetricTracker:
    """Helper class for tracking metrics during training"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.predictions = []
        self.labels = []
        self.masks = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        loss: float
    ):
        """Update tracked metrics with new batch"""
        self.predictions.append(predictions.detach())
        self.labels.append(labels.detach())
        self.masks.append(masks.detach())
        self.losses.append(loss)
    
    def compute(self, window_size: int = 10) -> Metrics:
        """Compute metrics from all tracked data"""
        predictions = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        masks = torch.cat(self.masks)
        avg_loss = np.mean(self.losses)
        
        return calculate_metrics(
            predictions, labels, masks, avg_loss, window_size
        ) 