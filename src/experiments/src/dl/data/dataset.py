import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
import sys

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from src.experiments.src.dl.data.processor import ProcessedMatch
from src.experiments.src.dl.config import DataConfig, TrainingConfig

class DotaMatchDataset(Dataset):

    def __init__(self, matches: List[ProcessedMatch], config: DataConfig):
        self.matches = self._deduplicate_matches(matches)
        self.config = config
        
    def __len__(self) -> int:
        return len(self.matches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        match = self.matches[idx]
        
        # Convert features to tensor
        features = torch.FloatTensor(match.features)
        
        # Create mask for valid timesteps (1 for valid, 0 for padding)
        mask = torch.zeros(self.config.max_game_length, dtype=torch.float)
        mask[:match.game_length] = 1.0
        
        # Create label tensor (same label for all timesteps)
        label = torch.full((self.config.max_game_length,), match.label, dtype=torch.float)
        
        return features, mask, label
    
    def _deduplicate_matches(self, matches: List[ProcessedMatch]) -> List[ProcessedMatch]:
            """Deduplicate matches by match_id while preserving order and print difference"""
            seen_ids = set()
            unique_matches = []
            for match in matches:
                if match.match_id not in seen_ids:
                    seen_ids.add(match.match_id)
                    unique_matches.append(match)
            
            original_len = len(matches)
            deduped = sorted(unique_matches, key=lambda x: x.match_id)
            print(f"Deduplicated matches: {original_len} -> {len(deduped)} (removed {original_len - len(deduped)} duplicates)")
            return deduped

def create_dataloaders(
    matches: List[ProcessedMatch],
    data_config: DataConfig=None,
    training_config: TrainingConfig=None,
    shuffle: bool = True,
    predict: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        matches: List of processed matches
        data_config: Data configuration
        training_config: Training configuration
        shuffle: Whether to shuffle the data
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if predict:
        return DataLoader(
            DotaMatchDataset(matches, data_config),
            batch_size=1,
            shuffle=False
        )
    # Split into train and validation
    val_size = int(len(matches) * training_config.validation_split)
    train_size = len(matches) - val_size
    
    train_matches = matches[:train_size]
    val_matches = matches[train_size:]
    
    # Create datasets
    train_dataset = DotaMatchDataset(train_matches, data_config)
    val_dataset = DotaMatchDataset(val_matches, data_config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_time_window_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    window_size: int = 10
) -> List[Tuple[float, float]]:
    """
    Calculate accuracy and AUC for each time window
    
    Args:
        predictions: Model predictions (batch_size, seq_len)
        labels: True labels (batch_size, seq_len)
        masks: Valid timestep masks (batch_size, seq_len)
        window_size: Size of time windows in minutes
    
    Returns:
        List of (accuracy, auc) tuples for each time window
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    import numpy as np
    
    seq_len = predictions.shape[1]
    n_windows = seq_len // window_size
    
    window_metrics = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        
        # Get predictions and labels for this window
        window_preds = predictions[:, start_idx:end_idx].reshape(-1)
        window_labels = labels[:, start_idx:end_idx].reshape(-1)
        window_masks = masks[:, start_idx:end_idx].reshape(-1)
        
        # Filter out padded timesteps
        valid_idx = window_masks.bool()
        if not valid_idx.any():
            window_metrics.append((0.0, 0.0))
            continue
            
        window_preds = window_preds[valid_idx].cpu().numpy()
        window_labels = window_labels[valid_idx].cpu().numpy()
        
        # Calculate metrics
        try:
            auc = roc_auc_score(window_labels, window_preds)
        except ValueError:
            auc = 0.5  # Default to random chance if only one class
            
        acc = accuracy_score(window_labels, window_preds > 0.5)
        
        window_metrics.append((acc, auc))
    
    return window_metrics 