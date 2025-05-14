import sys
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from src.experiments.src.dl.config import ModelConfig

class WinPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # Calculate the size of LSTM output
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional mask tensor of shape (batch_size, seq_len)
        
        Returns:
            Predictions tensor of shape (batch_size, seq_len)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Apply dense layers to each timestep
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out = lstm_out.reshape(-1, lstm_out.size(-1))  # (batch_size * seq_len, hidden_size)
        predictions = self.dense(lstm_out)  # (batch_size * seq_len, 1)
        predictions = predictions.reshape(batch_size, seq_len)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            predictions = predictions * mask
        
        return predictions
    
    def predict_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get probability predictions"""
        self.eval()
        with torch.no_grad():
            return self.forward(x, mask)
    
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions"""
        probs = self.predict_proba(x, mask)
        return (probs > threshold).float()
    
    def save(self, path: str):
        """Save model state"""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, config: ModelConfig) -> 'WinPredictor':
        """Load model from saved state"""
        checkpoint = torch.load(path)
        model = cls(config)
        model.load_state_dict(checkpoint)
        return model 