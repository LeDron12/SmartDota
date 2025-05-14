import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import yaml
from dataclasses import dataclass
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataclasses.match import MatchData
from src.data.dataclasses.public_match import PublicMatchData
from src.data.dataclasses.draft import DraftItem, PickBanItem
from src.data.api.OpenDota.pro_matches_dataloader import ProMatchesDataloader
from src.data.api.OpenDota.public_matches_dataloader import PublicMatchesDataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DraftDataProcessor:
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the directory containing match data
        """
        self.data_path = Path(data_path)
        self.hero_stats = None
        self.processed_data = None
        
    def load_data(self) -> None:
        """Load match data from files."""
        logger.info("Loading match data...")
        
        # Load matches
        raw_matches = self._load_matches(self.data_path / "public_110000_7-34b-ALL", is_train=True)
        
        # Filter matches to ensure both teams are present
        matches = [
            match for match in raw_matches 
            if match.radiant_team is not None and match.dire_team is not None
        ]
        logger.info(f"Loaded {len(raw_matches)} matches, kept {len(matches)} after filtering for complete team data.")
        
        # Calculate hero statistics first
        self._calculate_hero_stats(matches)
        
        # Process all matches into features
        self.processed_data = self._process_matches(matches)
        if self.processed_data[0].empty:
            raise ValueError("No valid matches found after processing")
            
        # Split into train and validation sets
        split_idx = int(len(self.processed_data[0]) * 0.9)
        self.train_data = (
            self.processed_data[0].iloc[:split_idx],
            self.processed_data[1].iloc[:split_idx]
        )
        self.val_data = (
            self.processed_data[0].iloc[split_idx:],
            self.processed_data[1].iloc[split_idx:]
        )
        
        logger.info(f"Split into {len(self.train_data[0])} training matches and {len(self.val_data[0])} validation matches")
        
    def _load_matches(self, file_path: Path, is_train: bool) -> List[MatchData]:
        """Load matches using appropriate dataloader."""
        if is_train:
            dataloader = PublicMatchesDataloader(0, 0)
        else:
            dataloader = ProMatchesDataloader(0, 0)
            
        dataloader.load(path=file_path)
        return dataloader.data
    
    def _calculate_hero_stats(self, matches: List[PublicMatchData]) -> None:
        """Calculate statistics for each hero from the matches."""
        logger.info("Calculating hero statistics...")
        
        # Initialize hero statistics
        hero_stats = {}
        
        # First pass: count total matches and wins for each hero
        for match in matches:
            # Process Radiant team
            for hero_id in match.radiant_team:
                if hero_id not in hero_stats:
                    hero_stats[hero_id] = {'wins': 0, 'total': 0}
                hero_stats[hero_id]['total'] += 1
                if match.radiant_win:
                    hero_stats[hero_id]['wins'] += 1
            
            # Process Dire team
            for hero_id in match.dire_team:
                if hero_id not in hero_stats:
                    hero_stats[hero_id] = {'wins': 0, 'total': 0}
                hero_stats[hero_id]['total'] += 1
                if not match.radiant_win:
                    hero_stats[hero_id]['wins'] += 1
        
        # Calculate win rates and pick rates
        total_matches = len(matches)
        for hero_id, stats in hero_stats.items():
            stats['win_rate'] = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            stats['pick_rate'] = stats['total'] / total_matches
        
        self.hero_stats = hero_stats
        logger.info(f"Calculated statistics for {len(hero_stats)} heroes")
        
        # Log some statistics
        win_rates = [stats['win_rate'] for stats in hero_stats.values()]
        logger.info(f"Average hero win rate: {np.mean(win_rates):.3f}")
        logger.info(f"Min hero win rate: {min(win_rates):.3f}")
        logger.info(f"Max hero win rate: {max(win_rates):.3f}")
    
    def _create_team_features(self, heroes: List[int]) -> Dict[str, float]:
        """Create features for a team composition."""
        features = {
            'avg_win_rate': 0.0,
            'avg_pick_rate': 0.0,
            'min_win_rate': 1.0,
            'max_win_rate': 0.0,
            'win_rate_std': 0.0,
            'pick_rate_std': 0.0
        }
        
        if not heroes:
            return features
            
        # Get statistics for each hero in the team
        hero_stats = [self.hero_stats.get(hero_id, {
            'win_rate': 0.5,
            'pick_rate': 0.0
        }) for hero_id in heroes]
        
        # Calculate team features
        features['avg_win_rate'] = np.mean([stats['win_rate'] for stats in hero_stats])
        features['avg_pick_rate'] = np.mean([stats['pick_rate'] for stats in hero_stats])
        features['min_win_rate'] = min([stats['win_rate'] for stats in hero_stats])
        features['max_win_rate'] = max([stats['win_rate'] for stats in hero_stats])
        features['win_rate_std'] = np.std([stats['win_rate'] for stats in hero_stats])
        features['pick_rate_std'] = np.std([stats['pick_rate'] for stats in hero_stats])
        
        return features
    
    def _process_matches(self, matches: List[PublicMatchData]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process matches into composition features and win labels.
        
        Args:
            matches: List of PublicMatchData objects
            
        Returns:
            Tuple of (features_df, win_labels)
        """
        logger.info("Processing matches into composition features...")
        
        # Initialize lists for data
        features_list = []
        win_labels = []
        
        for match in matches:
            if not match.radiant_team or not match.dire_team:
                logger.warning(f"Match {match.match_id} has incomplete pick/ban data, skipping...")
                continue
                
            # Get hero picks
            radiant_heroes = match.radiant_team
            dire_heroes = match.dire_team
            
            # Validate team sizes
            if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
                logger.warning(f"Match {match.match_id} has invalid team sizes: Radiant={len(radiant_heroes)}, Dire={len(dire_heroes)}, skipping...")
                continue
            
            # Create team features
            radiant_features = self._create_team_features(radiant_heroes)
            dire_features = self._create_team_features(dire_heroes)
            
            # Create match features
            match_features = {
                f'radiant_{k}': v for k, v in radiant_features.items()
            }
            match_features.update({
                f'dire_{k}': v for k, v in dire_features.items()
            })
            
            # Add relative features
            match_features['win_rate_diff'] = radiant_features['avg_win_rate'] - dire_features['avg_win_rate']
            match_features['pick_rate_diff'] = radiant_features['avg_pick_rate'] - dire_features['avg_pick_rate']
            match_features['win_rate_std_diff'] = radiant_features['win_rate_std'] - dire_features['win_rate_std']
            match_features['pick_rate_std_diff'] = radiant_features['pick_rate_std'] - dire_features['pick_rate_std']
            
            features_list.append(match_features)
            win_labels.append(match.radiant_win)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        win_labels = pd.Series(win_labels)
        
        logger.info(f"Processed {len(features_df)} matches")
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Win labels shape: {win_labels.shape}")
        
        return features_df, win_labels
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Get processed training and validation data.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if self.processed_data is None:
            self.load_data()
        
        return (
            self.train_data[0],  # X_train
            self.train_data[1],  # y_train
            self.val_data[0],    # X_val
            self.val_data[1]     # y_val
        )