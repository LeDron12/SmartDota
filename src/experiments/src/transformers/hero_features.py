import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
import logging
from collections import Counter
from ..core.base import BaseTransformer

logger = logging.getLogger(__name__)

class HeroFeaturesTransformer(BaseTransformer):
    """Transformer for processing hero features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hero features transformer.
        
        Args:
            config: Dictionary containing transformer configuration
        """
        super().__init__(config)
        self.hero_mapping = self._load_hero_mapping()
        self.num_heroes = len(self.hero_mapping)
        self._feature_names = None
        
        # Initialize feature names immediately
        self._feature_names = (
            [f'radiant_{i}' for i in range(self.num_heroes)] +
            [f'dire_{i}' for i in range(self.num_heroes)]
        )
        logger.info(f"Initialized {len(self._feature_names)} feature names")
    
    def _load_hero_mapping(self) -> Dict[int, Dict[str, Any]]:
        """Load hero ID to localized name mapping from heroes.json."""
        logger.info("Loading hero mapping...")
        try:
            # Get absolute path to heroes.json
            project_root = Path(__file__).parent.parent.parent.parent.parent
            heroes_path = project_root / self.config.get('heroes_path', 'data/heroes.json')
            
            logger.info(f"Loading heroes from: {heroes_path}")
            with open(heroes_path, 'r') as f:
                heroes = json.load(f)
                # Create mapping from original hero IDs to sequential IDs
                hero_mapping = {}
                for idx, hero in enumerate(heroes):
                    hero_mapping[hero['id']] = {
                        'name': hero['localized_name'],
                        'sequential_id': idx
                    }
                logger.info(f"Loaded {len(hero_mapping)} heroes")
                # Log a few sample heroes to verify
                sample_heroes = list(hero_mapping.items())[:5]
                for hero_id, info in sample_heroes:
                    logger.info(f"Sample hero: {hero_id} -> {info['name']} (seq_id: {info['sequential_id']})")
                return hero_mapping
        except Exception as e:
            logger.error(f"Failed to load hero mapping: {e}")
            raise  # Re-raise the exception to prevent silent failures
    
    def _format_hero_info(self, hero_id: int) -> str:
        """Format hero ID with name if available."""
        hero_info = self.hero_mapping.get(hero_id, {'name': 'Unknown', 'sequential_id': -1})
        return f"{hero_id} ({hero_info['name']})"
    
    def _get_sequential_id(self, hero_id: int) -> int:
        """Get sequential ID for a hero."""
        return self.hero_mapping.get(hero_id, {'sequential_id': -1})['sequential_id']
    
    def _get_hero_name(self, hero_id: int) -> str:
        """Get hero name from ID."""
        return self.hero_mapping.get(hero_id, {'name': 'Unknown'})['name']
    
    def _analyze_hero_distribution(self, df: pd.DataFrame) -> None:
        """Analyze and log hero pick distribution."""
        all_heroes = []
        for _, row in df.iterrows():
            all_heroes.extend(row['radiant_team'])
            all_heroes.extend(row['dire_team'])
        
        hero_counts = Counter(all_heroes)
        most_common_heroes = [(h, c) for h, c in hero_counts.most_common(10)]
        least_common_heroes = [(h, c) for h, c in hero_counts.most_common()[:-11:-1]]
        
        logger.info(f"Found {len(set(all_heroes))} unique heroes")
        logger.info("Hero pick distribution:")
        logger.info("Most picked heroes:")
        for hero_id, count in most_common_heroes:
            logger.info(f"  {self._format_hero_info(hero_id)}: {count} picks")
        logger.info("Least picked heroes:")
        for hero_id, count in least_common_heroes:
            logger.info(f"  {self._format_hero_info(hero_id)}: {count} picks")
    
    def fit(self, df: pd.DataFrame) -> 'HeroFeaturesTransformer':
        """
        Fit the transformer to the data.
        
        Args:
            df: Input DataFrame with hero picks
            
        Returns:
            self for method chaining
        """
        # Analyze hero distribution
        self._analyze_hero_distribution(df)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform hero picks into feature vectors.
        
        Args:
            df: Input DataFrame with hero picks
            
        Returns:
            DataFrame with hero features
        """
        logger.info("Processing matches into hero ID features...")
        
        features = []
        # labels = []
        skipped_matches = 0
        
        for _, row in df.iterrows():
            if not row['radiant_team'] or not row['dire_team']:
                logger.warning(f"Match {row['match_id']} has incomplete pick/ban data, skipping...")
                skipped_matches += 1
                continue
            
            # Validate team sizes
            if len(row['radiant_team']) != 5 or len(row['dire_team']) != 5:
                logger.warning(f"Match {row['match_id']} has invalid team sizes: Radiant={len(row['radiant_team'])}, Dire={len(row['dire_team'])}, skipping...")
                skipped_matches += 1
                continue
            
            # Create feature vector
            feature_vector = np.zeros(2 * self.num_heroes, dtype=np.int8)  # Use int8 for OHE
            
            # Set Radiant heroes (first half of vector)
            for hero_id in row['radiant_team']:
                seq_id = self._get_sequential_id(hero_id)
                if seq_id >= 0:  # Only process valid hero IDs
                    feature_vector[seq_id] = 1
            
            # Set Dire heroes (second half of vector)
            for hero_id in row['dire_team']:
                seq_id = self._get_sequential_id(hero_id)
                if seq_id >= 0:  # Only process valid hero IDs
                    feature_vector[self.num_heroes + seq_id] = 1
            
            features.append(feature_vector)
            # labels.append(1 if row['radiant_win'] else 0)
        
        # Convert to DataFrame with integer dtype
        features_df = pd.DataFrame(
            features,
            columns=self._feature_names,
            dtype=np.int8  # Ensure integer dtype for OHE features
        )
        # win_labels = pd.Series(labels, dtype=np.int8)
        
        # # Add win labels to the DataFrame
        # features_df['radiant_win'] = win_labels
        
        logger.info(f"Processed {len(features_df)} matches")
        logger.info(f"Skipped {skipped_matches} matches due to invalid data")
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Feature dtype: {features_df.dtypes.iloc[0]}")
        
        return features_df
    
    def get_feature_names(self, categorical: bool = False) -> List[str]:
        """
        Get the names of the features after transformation.
        
        Returns:
            List of feature names
        """
        if categorical:
            # return self._feature_names if self._feature_names else []
            return []
        return self._feature_names if self._feature_names else []