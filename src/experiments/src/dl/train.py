import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict
import argparse
from pathlib import Path

sys.path.append("data/")
sys.path.append("models/")
sys.path.append("training/")

from config import Config
from data.processor import MatchDataProcessor, ProcessedMatch
from data.dataset import create_dataloaders
from models.win_predictor import WinPredictor
from training.trainer import Trainer

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_matches(data_path: str) -> List[Dict]:
    """Load match data from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def main(args):
    # Create config
    config = Config()
    
    # Override config with command line arguments
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    
    # Set random seed
    set_seed(config.training.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading matches...")
    matches = load_matches(args.data_path)
    
    print("Processing matches...")
    processor = MatchDataProcessor(config.data)
    processed_matches = processor.process_matches(matches)
    
    # Save feature scalers
    # processor.save_scalers(os.path.join(args.output_dir, 'feature_scalers.json'))
    
    print(f"Processed {len(processed_matches)} matches")
    # print(processed_matches[100])
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        processed_matches,
        config.data,
        config.training
    )

    # for elem in train_loader:
    #     print('train_loader_elem', elem)
    #     print('train_loader_elem[0]', elem[0], elem[0].shape)
    #     print('train_loader_elem[1]', elem[1], elem[1].shape)
    #     print('train_loader_elem[2]', elem[2], elem[2].shape)
    #     break
    
    # Create model
    print("Creating model...")
    model = WinPredictor(config.model)
    
    # Create trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir=args.output_dir
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dota 2 win prediction model")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to JSON file containing match data")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save model and logs")
    parser.add_argument("--learning_rate", type=float,
                      help="Learning rate (overrides config)")
    parser.add_argument("--batch_size", type=int,
                      help="Batch size (overrides config)")
    parser.add_argument("--num_epochs", type=int,
                      help="Number of epochs (overrides config)")
    
    args = parser.parse_args()
    main(args) 