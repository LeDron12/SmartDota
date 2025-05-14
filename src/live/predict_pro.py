import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import argparse
from datetime import datetime

# Add the experiments directory to Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'src', 'dl'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_new.fetch_matches import fetch_opendota_matches

from src.experiments.src.dl.models.win_predictor import WinPredictor
from src.experiments.src.dl.data.processor import MatchDataProcessor
from src.experiments.src.dl.data.dataset import create_dataloaders
from src.experiments.src.dl.config import Config, DataConfig, ModelConfig

class MatchPredictor:
    def __init__(self, model_path: str, config: ModelConfig, data_config: DataConfig):
        """
        Initialize predictor with a trained model
        
        Args:
            model_path: Path to the model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path, config)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.data_config = data_config
    
    def _load_model(self, model_path: str, config: ModelConfig) -> WinPredictor:
        """Load model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        return WinPredictor.load(model_path, config)
    
    def load_match_data(self, match_id: int, processor: MatchDataProcessor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process match data
        
        Args:
            match_id: Dota 2 match ID
            
        Returns:
            Tuple of (features, mask) tensors
            - features: shape (1, seq_len, n_features)
            - mask: shape (1, seq_len)
        """
        match_data, _ = fetch_opendota_matches('match_data', {'match_id': match_id, 'start_time_start': 1743800400}, full_path=True)
        match_data = processor.process_matches(match_data['rows'])
        model_data = create_dataloaders(match_data, data_config=self.data_config, predict=True)
        features, mask, _ = next(iter(model_data))

        return features, mask

    
    def predict(self, match_id: int, processor: MatchDataProcessor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate predictions for a match
        
        Args:
            match_id: Dota 2 match ID
            
        Returns:
            Tuple of (minutes, probabilities, true_label)
            - minutes: array of game minutes
            - probabilities: array of win probabilities
            - true_label: true match outcome (1 for Radiant win, 0 for Dire win)
        """
        # Load match data
        print(match_id, processor)
        features, mask = self.load_match_data(match_id, processor)
        
        # Move to device
        features = features.to(self.device)
        mask = mask.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            probs = self.model.predict_proba(features, mask)
        
        # Convert to numpy and get valid timesteps
        probs = probs.cpu().numpy()
        mask = mask.cpu().numpy()
        
        # Get minutes where mask is valid
        minutes = np.arange(len(mask[0]))[mask[0].astype(bool)]
        probabilities = probs[0][mask[0].astype(bool)]
        
        # TODO: Get true label from match data
        true_label = None  # This should be set based on the actual match outcome
        
        return minutes, probabilities, true_label
    
    def plot_predictions(
        self,
        match_id: int,
        processor: MatchDataProcessor,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Generate and save prediction plot
        
        Args:
            match_id: Dota 2 match ID
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to the saved plot
        """
        minutes, probabilities, true_label = self.predict(match_id, processor)
        
        plt.figure(figsize=(12, 6))
        plt.plot(minutes, probabilities, 'b-', label='Win Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')
        if true_label is not None:
            plt.axhline(y=true_label, color='g', linestyle=':', label='True Outcome')
        
        plt.title(f'Win Probability Over Time (Match {match_id})')
        plt.xlabel('Game Minute')
        plt.ylabel('Radiant Win Probability')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            save_path = os.path.join(
                os.path.dirname(__file__),
                f'predictions_{match_id}_{timestamp}.png'
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path

def main():
    parser = argparse.ArgumentParser(description='Generate win probability predictions for a Dota 2 match')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--match_id', type=int, required=True, help='Dota 2 match ID')
    parser.add_argument('--show_plot', action='store_true', help='Display the plot')
    
    args = parser.parse_args()
    
    # try:
    config_path = Path(args.model_path).parent / 'config.json'
    model_config = ModelConfig.from_json(str(config_path))
    data_config = DataConfig.from_json(str(config_path))

    predictor = MatchPredictor(args.model_path, model_config, data_config)
    processor = MatchDataProcessor(data_config)
    plot_path = predictor.plot_predictions(args.match_id, processor, show_plot=args.show_plot)
    print(f"\nPrediction plot saved to: {plot_path}")
# except Exception as e:
#     print(f"Error generating predictions: {str(e)}")
#     sys.exit(1)

if __name__ == '__main__':
    main() 