import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import yaml
import sys
import os
from collections import Counter
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataclasses.match import MatchData
from src.data.dataclasses.public_match import PublicMatchData
from src.data.api.OpenDota.public_matches_dataloader import PublicMatchesDataloader

logger = logging.getLogger(__name__)

# Load hero mapping
def load_hero_mapping() -> Dict[int, str]:
    """Load hero ID to localized name mapping from heroes.json."""
    logger.info("Loading hero mapping...")
    try:
        heroes_path = project_root / 'data' / 'heroes.json'
        with open(heroes_path, 'r') as f:
            heroes = json.load(f)
            # Create mapping from original hero IDs to sequential IDs (0 to len(heroes)-1)
            hero_mapping = {}
            for idx, hero in enumerate(heroes):
                hero_mapping[hero['id']] = {
                    'name': hero['localized_name'],
                    'sequential_id': idx
                }
            logger.info(f"Loaded {len(hero_mapping)} heroes")
            return hero_mapping
    except Exception as e:
        logger.warning(f"Failed to load hero mapping: {e}")
        return {}

# Load hero mapping at module level
HERO_MAPPING = load_hero_mapping()

def format_hero_info(hero_id: int) -> str:
    """Format hero ID with name if available."""
    hero_info = HERO_MAPPING.get(hero_id, {'name': 'Unknown', 'sequential_id': -1})
    return f"{hero_id} ({hero_info['name']})"

def get_sequential_id(hero_id: int) -> int:
    """Get sequential ID for a hero."""
    return HERO_MAPPING.get(hero_id, {'sequential_id': -1})['sequential_id']

def get_hero_name(hero_id: int) -> str:
    """Get hero name from ID."""
    return HERO_MAPPING.get(hero_id, {'name': 'Unknown'})['name']

class DraftDataProcessor:
    def __init__(self, data_path: Path):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.processed_data = None
        self.num_heroes = len(HERO_MAPPING)  # Number of unique heroes
        self.train_data = None
        self.val_data = None
    
    def _save_sample_data(self) -> None:
        """Save sample dataframes as CSV and Excel files."""
        logger.info("Saving sample data...")
        
        # Create output directory
        output_dir = Path(__file__).parent / 'input_sample'
        output_dir.mkdir(exist_ok=True)
        
        # Get 20 samples from each dataset
        train_samples = self.train_data[0].head(20)
        train_labels = self.train_data[1].head(20)
        val_samples = self.val_data[0].head(20)
        val_labels = self.val_data[1].head(20)
        
        # Create more readable column names
        def get_readable_columns(df):
            readable_cols = {}
            for col in df.columns:
                team, idx = col.split('_')
                hero_id = next((h_id for h_id, info in HERO_MAPPING.items() 
                              if info['sequential_id'] == int(idx)), -1)
                hero_name = get_hero_name(hero_id)
                readable_cols[col] = f"{team}_{hero_name}"
            return readable_cols
        
        # Save training data
        train_df = train_samples.copy()
        train_df.columns = [get_readable_columns(train_df)[col] for col in train_df.columns]
        train_df['radiant_win'] = train_labels
        
        # Save validation data
        val_df = val_samples.copy()
        val_df.columns = [get_readable_columns(val_df)[col] for col in val_df.columns]
        val_df['radiant_win'] = val_labels
        
        # Save as CSV
        train_df.to_csv(output_dir / 'train_sample.csv', index=False)
        val_df.to_csv(output_dir / 'val_sample.csv', index=False)
        
        # Save as Excel
        with pd.ExcelWriter(output_dir / 'input_samples.xlsx') as writer:
            train_df.to_excel(writer, sheet_name='Train Sample', index=False)
            val_df.to_excel(writer, sheet_name='Val Sample', index=False)
        
        logger.info(f"Saved sample data to {output_dir}")
        logger.info(f"Training sample shape: {train_df.shape}")
        logger.info(f"Validation sample shape: {val_df.shape}")

    def load_data(self) -> None:
        """Load and process match data."""
        logger.info("Loading match data...")
        
        # Load matches
        raw_matches = self._load_matches(self.data_path / "public_110000_7-34b-ALL", is_train=True)
        
        # Log sample raw match
        if raw_matches:
            sample_match = raw_matches[0]
            logger.info("Sample raw match data:")
            logger.info(f"Match ID: {sample_match.match_id}")
            logger.info(f"Radiant team: {[format_hero_info(h) for h in sample_match.radiant_team]}")
            logger.info(f"Dire team: {[format_hero_info(h) for h in sample_match.dire_team]}")
            logger.info(f"Radiant win: {sample_match.radiant_win}")
        
        # Filter matches to ensure both teams are present
        matches = [
            match for match in raw_matches 
            if match.radiant_team is not None and match.dire_team is not None
        ]
        logger.info(f"Loaded {len(raw_matches)} matches, kept {len(matches)} after filtering for complete team data.")
        
        # Analyze hero distribution
        all_heroes = []
        for match in matches:
            all_heroes.extend(match.radiant_team)
            all_heroes.extend(match.dire_team)
        
        # Analyze hero distribution
        hero_counts = Counter(all_heroes)
        most_common_heroes = [(h, c) for h, c in hero_counts.most_common(10)]
        least_common_heroes = [(h, c) for h, c in hero_counts.most_common()[:-11:-1]]
        
        logger.info(f"Found {len(set(all_heroes))} unique heroes")
        logger.info("Hero pick distribution:")
        logger.info("Most picked heroes:")
        for hero_id, count in most_common_heroes:
            logger.info(f"  {format_hero_info(hero_id)}: {count} picks")
        logger.info("Least picked heroes:")
        for hero_id, count in least_common_heroes:
            logger.info(f"  {format_hero_info(hero_id)}: {count} picks")
        
        # Process matches
        self.processed_data = self._process_matches(matches)
        if self.processed_data[0].empty:
            raise ValueError("No valid matches found after processing")
        
        # Log sample processed data
        logger.info("\nSample processed data:")
        sample_features = self.processed_data[0].iloc[0]
        sample_label = self.processed_data[1].iloc[0]
        logger.info(f"Sample feature vector shape: {sample_features.shape}")
        non_zero_indices = sample_features[sample_features != 0].index.tolist()
        logger.info("Sample feature vector non-zero elements:")
        for idx in non_zero_indices:
            team, hero_idx = idx.split('_')
            hero_idx = int(hero_idx)
            # Find original hero ID from sequential ID
            original_hero_id = next((h_id for h_id, info in HERO_MAPPING.items() 
                                   if info['sequential_id'] == hero_idx), -1)
            logger.info(f"  {team} hero: {format_hero_info(original_hero_id)}")
        logger.info(f"Sample label: {sample_label}")
        
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
        
        # Log data split statistics
        logger.info("\nData split statistics:")
        logger.info(f"Training set: {len(self.train_data[0])} matches")
        logger.info(f"Validation set: {len(self.val_data[0])} matches")
        logger.info(f"Training set win rate: {self.train_data[1].mean():.2%}")
        logger.info(f"Validation set win rate: {self.val_data[1].mean():.2%}")
        
        # Log feature statistics
        logger.info("\nFeature statistics:")
        logger.info(f"Total features: {self.processed_data[0].shape[1]}")
        logger.info(f"Features per team: {self.num_heroes}")
        logger.info(f"Average non-zero features per match: {self.processed_data[0].astype(bool).sum(axis=1).mean():.2f}")
        
        # Save sample data
        self._save_sample_data()
        
    def _load_matches(self, file_path: Path, is_train: bool) -> List[MatchData]:
        """Load matches using appropriate dataloader."""
        logger.info(f"Loading matches from: {file_path}")
        dataloader = PublicMatchesDataloader(0, 0)
        dataloader.load(path=file_path)
        return dataloader.data
    
    def _process_matches(self, matches: List[PublicMatchData]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process matches into features and labels.
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            Tuple of (features_df, win_labels)
        """
        logger.info("Processing matches into hero ID features...")
        
        features = []
        labels = []
        skipped_matches = 0
        
        for match in matches:
            if not match.radiant_team or not match.dire_team:
                logger.warning(f"Match {match.match_id} has incomplete pick/ban data, skipping...")
                skipped_matches += 1
                continue
            
            # Validate team sizes
            if len(match.radiant_team) != 5 or len(match.dire_team) != 5:
                logger.warning(f"Match {match.match_id} has invalid team sizes: Radiant={len(match.radiant_team)}, Dire={len(match.dire_team)}, skipping...")
                skipped_matches += 1
                continue
            
            # Create feature vector
            feature_vector = self._create_hero_features(match)
            features.append(feature_vector)
            
            # Create label (1 for Radiant win, 0 for Dire win)
            labels.append(1 if match.radiant_win else 0)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(
            features,
            columns=[f'radiant_{i}' for i in range(self.num_heroes)] + 
                    [f'dire_{i}' for i in range(self.num_heroes)]
        )
        win_labels = pd.Series(labels)
        
        logger.info(f"Processed {len(features_df)} matches")
        logger.info(f"Skipped {skipped_matches} matches due to invalid data")
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Win labels shape: {win_labels.shape}")
        
        return features_df, win_labels
    
    def _create_hero_features(self, match: PublicMatchData) -> np.ndarray:
        """
        Create feature vector from hero IDs.
        
        Args:
            match: Match object containing hero picks
            
        Returns:
            Feature vector of size 2 * num_heroes
            First half represents Radiant heroes, second half represents Dire heroes
        """
        # Initialize feature vector
        feature_vector = np.zeros(2 * self.num_heroes)
        
        # Set Radiant heroes (first half of vector)
        for hero_id in match.radiant_team:
            seq_id = get_sequential_id(hero_id)
            if seq_id >= 0:  # Only process valid hero IDs
                feature_vector[seq_id] = 1
        
        # Set Dire heroes (second half of vector)
        for hero_id in match.dire_team:
            seq_id = get_sequential_id(hero_id)
            if seq_id >= 0:  # Only process valid hero IDs
                feature_vector[self.num_heroes + seq_id] = 1
        
        return feature_vector
    
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