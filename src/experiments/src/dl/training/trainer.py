import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Tuple
import os
import sys
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from src.experiments.src.dl.models.win_predictor import WinPredictor
from src.experiments.src.dl.config import Config
from src.experiments.src.dl.training.metrics import Metrics, MetricTracker

class Trainer:
    def __init__(
        self,
        model: WinPredictor,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.training.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        self.criterion = nn.BCELoss(reduction='none')  # Will handle masking manually
        
        # Setup logging with timestamped subfolder
        if log_dir is None:
            log_dir = config.logging.log_dir
        
        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.log_dir = os.path.join(
            log_dir,
            f"{config.logging.experiment_name}_{timestamp}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.epoch = 0
        
        print(f"\nSaving outputs to: {self.log_dir}")
    
    def train_epoch(self) -> Metrics:
        """Train for one epoch"""
        self.model.train()
        metric_tracker = MetricTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (features, masks, labels) in enumerate(pbar):
            # Move data to device
            features = features.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            predictions = self.model(features, masks)
            
            # Calculate loss (only on valid timesteps)
            loss = self.criterion(predictions, labels)
            loss = (loss * masks).sum() / masks.sum()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metric_tracker.update(predictions, labels, masks, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.writer.add_scalar('train/loss', loss.item(), 
                                     self.epoch * len(self.train_loader) + batch_idx)
        
        # Compute and log epoch metrics
        metrics = metric_tracker.compute(self.config.data.time_window_size)
        self._log_metrics(metrics, 'train')
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Metrics:
        """Validate the model"""
        self.model.eval()
        metric_tracker = MetricTracker()
        
        with torch.inference_mode():
            for features, masks, labels in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                features = features.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                predictions = self.model(features, masks)
                
                # Calculate loss
                loss = self.criterion(predictions, labels)
                loss = (loss * masks).sum() / masks.sum()
                
                # Update metrics
                metric_tracker.update(predictions, labels, masks, loss.item())
            
            # Compute and log validation metrics
            metrics = metric_tracker.compute(self.config.data.time_window_size)
            self._log_metrics(metrics, 'val')
        
        return metrics
    
    def _log_metrics(self, metrics: Metrics, prefix: str):
        """Log metrics to tensorboard"""
        metrics_dict = metrics.to_dict()
        
        # Log overall metrics
        for name, value in metrics_dict.items():
            if name != 'window_metrics':
                self.writer.add_scalar(f'{prefix}/{name}', value, self.epoch)
        
        # Log windowed metrics
        for i, window_metrics in enumerate(metrics_dict['window_metrics']):
            for name, value in window_metrics.items():
                self.writer.add_scalar(
                    f'{prefix}/window_{i+1}/{name}',
                    value,
                    self.epoch
                )
    
    def predict_match(self, match: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a single match
        
        Args:
            match: Tuple of (features, mask, label) tensors
            
        Returns:
            Tuple of (minutes, probabilities) arrays
        """
        self.model.eval()
        features, mask, _ = match
        
        # Move to device
        features = features.to(self.device)
        mask = mask.to(self.device)
        
        # Get predictions
        with torch.inference_mode():
            probs = self.model.predict_proba(features, mask)
        
        # Convert to numpy and get valid timesteps
        probs = probs.cpu().numpy()
        mask = mask.cpu().numpy()
        
        # Get minutes where mask is valid
        minutes = np.arange(len(mask[0]))[mask[0].astype(bool)]
        probabilities = probs[0][mask[0].astype(bool)]
        
        return minutes, probabilities
    
    def plot_match_predictions(self, match: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], save_path: Optional[str] = None):
        """
        Plot win probability predictions for a match over time
        
        Args:
            match: Tuple of (features, mask, label) tensors
            save_path: Optional path to save the plot
        """
        minutes, probabilities = self.predict_match(match)
        # Get true label for the first valid timestep
        mask = match[1][0].cpu().numpy()
        labels = match[2][0].cpu().numpy()
        first_valid_idx = np.where(mask)[0][0]
        true_label = labels[first_valid_idx]  # Get label at first valid timestep
        
        plt.figure(figsize=(12, 6))
        plt.plot(minutes, probabilities, 'b-', label='Win Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')
        plt.axhline(y=true_label, color='g', linestyle=':', label='True Outcome')
        
        plt.title('Win Probability Over Time')
        plt.xlabel('Game Minute')
        plt.ylabel('Radiant Win Probability')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def train(self):
        """Train the model"""
        print("\nStarting training...")
        print("=" * 80)
        print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train AUC':^10} | {'Val Loss':^10} | {'Val AUC':^10} | {'Best Val AUC':^12} | {'Window':^6}")
        print("-" * 80)

        best_metrics = {
            'epoch': 0,
            'train': None,
            'val': None,
            'window_metrics': None
        }
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Print metrics
            print(f"{epoch:^6d} | {train_metrics.loss:^10.4f} | {train_metrics.auc:^10.4f} | "
                  f"{val_metrics.loss:^10.4f} | {val_metrics.auc:^10.4f} | {max(self.best_val_auc, val_metrics.auc):^12.4f} | "
                  f"{self.config.data.time_window_size:^6d}")
            
            print("\nWindowed Validation Metrics:")
            print("-" * 40)
            print(f"{'Window':^8} | {'Accuracy':^10} | {'AUC':^10}")
            print("-" * 40)
            for i, (acc, auc) in enumerate(val_metrics.window_metrics):
                print(f"{i+1:^8d} | {acc:^10.4f} | {auc:^10.4f}")
            print("-" * 40 + "\n")
            
            # Check for early stopping
            if val_metrics.auc > self.best_val_auc:
                self.best_val_auc = val_metrics.auc
                self.patience_counter = 0
                print(f"\nNew best validation AUC: {val_metrics.auc:.4f}")
                
                # Save best model
                self.model.save(os.path.join(self.log_dir, 'best_model.pt'))
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.logging.save_every_n_epochs == 0:
                self.model.save(os.path.join(self.log_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        print("\nTraining completed!")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")

        # Update best metrics
        best_metrics.update({
            'epoch': epoch,
            'train': train_metrics.to_dict(),
            'val': val_metrics.to_dict(),
            'window_metrics': [{'accuracy': acc, 'auc': auc} for acc, auc in val_metrics.window_metrics]
        })
        
        # Save final model and plot
        self.model.save(os.path.join(self.log_dir, 'final_model.pt'))
        for i, val_match in enumerate(self.val_loader):
            if i == 10:
                break
            plot_path = os.path.join(self.log_dir, f'final_model_predictions_{i+1}.png')
            self.plot_match_predictions(val_match, save_path=plot_path)
            print(f"Saved final prediction plot to: {plot_path}")

        # Save final metrics
        final_metrics = {
            'best_metrics': best_metrics,
            'final_epoch': epoch,
            'early_stopped': self.patience_counter >= self.config.training.early_stopping_patience,
            'best_val_auc': self.best_val_auc,
            'config': self.config.to_dict()
        }
        
        metrics_path = os.path.join(self.log_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"\nSaved training metrics to: {metrics_path}")
        
        self.writer.close()
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.patience_counter = checkpoint['patience_counter'] 