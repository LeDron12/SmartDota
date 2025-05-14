from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import json

@dataclass
class ModelConfig:
    input_size: int =  17  # Number of features per timestep [2, 6]
    hidden_size: int = 64  # Increased from 64
    num_layers: int = 1  # Increased from 1
    dropout: float = 0.2  # Reduced from 0.2
    bidirectional: bool = True  # Changed to True for better feature extraction

    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)["model"]
        return cls(**config_dict)

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 3  # Increased from 10
    learning_rate: float = 0.001  # Reduced from 0.01
    weight_decay: float = 1e-4  # Increased from 1e-5 for better regularization
    early_stopping_patience: int = 2  # Increased from 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    validation_split: float = 0.1  # Increased from 0.05
    random_seed: int = 42

@dataclass
class DataConfig:
    max_game_length: int = 60  # Maximum game length in minutes
    min_game_length: int = 10  # Minimum game length to consider
    time_window_size: int = 10  # Size of time windows for validation
    feature_columns: List[str] = None

    @classmethod
    def from_json(cls, json_path: str) -> 'DataConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)["data"]
        return cls(**config_dict)
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'gold_advantage',
                'xp_advantage',
                'radiant_towers',
                'dire_towers',
                'first_blood',
                'last_roshan_killed',
                'radiant_damage',
                'dire_damage',
                'radiant_healing',
                'dire_healing',
                'radiant_deaths',
                'dire_deaths',
                'radiant_gold_delta',
                'dire_gold_delta',
                'radiant_xp_delta',
                'dire_xp_delta',
                'teamfight_count'
            ]

@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    experiment_name: str = "win_predictor"
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 5
    tensorboard: bool = True

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__
        } 