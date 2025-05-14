import logging
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
from ..core.base import BaseTransformer

logger = logging.getLogger(__name__)

class HeroStatsTransformer(BaseTransformer):
    """
    Transformer for calculating hero statistics features from Dota 2 match data.
    
    This transformer calculates various hero-based statistics including:
    1. Hero winrates within team
    2. Hero with Hero winrates within team (synergy)
    3. Hero with Hero winrates versus enemy team (counters)
    4. Hero pickrates within team
    5. Hero banrates within team
    
    Each statistic is calculated for both Radiant and Dire teams using historical match data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer.
        
        Args:
            config: Configuration dictionary containing:
                - heroes_path: Path to the heroes.json file
        """
        super().__init__(config)
        self.heroes_path = config.get('heroes_path')
        self.exclude_features = set(config.get('exclude_features', []))
        logger.info(f"Excluding features: {self.exclude_features}")

        self.heroes_data = None
        self.hero_stats = None
        self.feature_names = []
        
    def _load_heroes_data(self) -> None:
        """Load hero data from heroes.json file."""
        try:
            # Get absolute path to heroes.json
            project_root = Path(__file__).parent.parent.parent.parent.parent
            heroes_path = project_root / self.heroes_path
            
            logger.info(f"Loading heroes from: {heroes_path}")
            with open(heroes_path, 'r') as f:
                self.heroes_data = json.load(f)
            logger.info(f"Loaded hero data from {heroes_path}")
            logger.info(f"Number of heroes: {len(self.heroes_data)}")
        except Exception as e:
            logger.error(f"Failed to load hero data: {str(e)}")
            raise
            
    def _calculate_hero_stats(self, X: pd.DataFrame) -> None:
        """
        Calculate hero statistics from match data.
        
        Args:
            X: Input DataFrame with match data
        """
        logger.info("Calculating hero statistics from match data...")
        
        # Initialize statistics dictionaries
        hero_stats = {
            'matches_played': {},  # Total matches played
            'wins': {},           # Total wins
            'picks': {},          # Total picks
            # 'bans': {},           # Total bans
            'with_hero': {},      # Matches played with each other hero
            'with_hero_wins': {}, # Wins when played with each other hero
            'vs_hero': {},        # Matches played against each hero
            'vs_hero_wins': {}    # Wins when played against each hero
        }
        
        # Initialize counters for each hero
        for hero in self.heroes_data:
            hero_id = hero['id']
            hero_stats['matches_played'][hero_id] = 0
            hero_stats['wins'][hero_id] = 0
            hero_stats['picks'][hero_id] = 0
            # hero_stats['bans'][hero_id] = 0
            hero_stats['with_hero'][hero_id] = {}
            hero_stats['with_hero_wins'][hero_id] = {}
            hero_stats['vs_hero'][hero_id] = {}
            hero_stats['vs_hero_wins'][hero_id] = {}
            
            # Initialize pairwise counters
            for other_hero in self.heroes_data:
                other_id = other_hero['id']
                hero_stats['with_hero'][hero_id][other_id] = 0
                hero_stats['with_hero_wins'][hero_id][other_id] = 0
                hero_stats['vs_hero'][hero_id][other_id] = 0
                hero_stats['vs_hero_wins'][hero_id][other_id] = 0
        
        # Process each match
        for _, match in X.iterrows():
            radiant_heroes = match['radiant_team']
            dire_heroes = match['dire_team']
            radiant_win = match['radiant_win']
            
            # Process Radiant team
            for hero_id in radiant_heroes:
                hero_stats['matches_played'][hero_id] += 1
                hero_stats['picks'][hero_id] += 1
                if radiant_win:
                    hero_stats['wins'][hero_id] += 1
                
                # Update synergy stats
                for other_hero_id in radiant_heroes:
                    if hero_id != other_hero_id:
                        hero_stats['with_hero'][hero_id][other_hero_id] += 1
                        if radiant_win:
                            hero_stats['with_hero_wins'][hero_id][other_hero_id] += 1
                
                # Update counter stats
                for enemy_hero_id in dire_heroes:
                    hero_stats['vs_hero'][hero_id][enemy_hero_id] += 1
                    if radiant_win:
                        hero_stats['vs_hero_wins'][hero_id][enemy_hero_id] += 1
            
            # Process Dire team
            for hero_id in dire_heroes:
                hero_stats['matches_played'][hero_id] += 1
                hero_stats['picks'][hero_id] += 1
                if not radiant_win:
                    hero_stats['wins'][hero_id] += 1
                
                # Update synergy stats
                for other_hero_id in dire_heroes:
                    if hero_id != other_hero_id:
                        hero_stats['with_hero'][hero_id][other_hero_id] += 1
                        if not radiant_win:
                            hero_stats['with_hero_wins'][hero_id][other_hero_id] += 1
                
                # Update counter stats
                for enemy_hero_id in radiant_heroes:
                    hero_stats['vs_hero'][hero_id][enemy_hero_id] += 1
                    if not radiant_win:
                        hero_stats['vs_hero_wins'][hero_id][enemy_hero_id] += 1
        
        # Calculate final statistics
        self.hero_stats = {
            'winrate': {},
            'pickrate': {},
            # 'banrate': {},
            'with_hero_winrate': {},
            'vs_hero_winrate': {}
        }
        
        total_matches = len(X)
        
        for hero_id in hero_stats['matches_played']:
            # Winrate
            matches = hero_stats['matches_played'][hero_id]
            wins = hero_stats['wins'][hero_id]
            self.hero_stats['winrate'][hero_id] = wins / matches if matches > 0 else 0.5 # might fill avg winrate if matches == 0
            
            # Pickrate
            picks = hero_stats['picks'][hero_id]
            self.hero_stats['pickrate'][hero_id] = picks / total_matches if total_matches > 0 else 0
            
            # # Banrate (if available in data)
            # bans = hero_stats['bans'][hero_id]
            # self.hero_stats['banrate'][hero_id] = bans / total_matches if total_matches > 0 else 0
            
            # With hero winrates
            self.hero_stats['with_hero_winrate'][hero_id] = {}
            for other_id in hero_stats['with_hero'][hero_id]:
                with_matches = hero_stats['with_hero'][hero_id][other_id]
                with_wins = hero_stats['with_hero_wins'][hero_id][other_id]
                self.hero_stats['with_hero_winrate'][hero_id][other_id] = (
                    with_wins / with_matches if with_matches > 0 else 0.5 # might fill avg winrate if matches == 0
                )
            
            # Versus hero winrates
            self.hero_stats['vs_hero_winrate'][hero_id] = {}
            for other_id in hero_stats['vs_hero'][hero_id]:
                vs_matches = hero_stats['vs_hero'][hero_id][other_id]
                vs_wins = hero_stats['vs_hero_wins'][hero_id][other_id]
                self.hero_stats['vs_hero_winrate'][hero_id][other_id] = (
                    vs_wins / vs_matches if vs_matches > 0 else 0.5 # might fill avg winrate if matches == 0
                )
        
        logger.info("Finished calculating hero statistics")
        
    def _calculate_hero_winrates(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate hero winrates within team."""
        logger.info("Calculating hero winrates within team...")
        feature_names = []

        for team in ['radiant', 'dire']:
            team_col = f'{team}_team'
            
            # Calculate team average winrate
            avg_feature = f'{team}_avg_hero_winrate'
            X[avg_feature] = X.apply(
                lambda row: np.mean([
                    self.hero_stats['winrate'][hero_id]
                    for hero_id in row[team_col]
                ]),
                axis=1
            )
            feature_names.append(avg_feature)
        
        return X, feature_names
    
    def _calculate_hero_with_hero_winrates(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate hero with hero winrates within team."""
        logger.info("Calculating hero with hero winrates within team...")
        feature_names = []
        
        for team in ['radiant', 'dire']:
            team_col = f'{team}_team'
            
            # Calculate team average synergy
            avg_feature = f'{team}_avg_hero_synergy'
            X[avg_feature] = X.apply(
                lambda row: np.mean([
                    self.hero_stats['with_hero_winrate'][hero1][hero2]
                    for i, hero1 in enumerate(row[team_col])
                    for hero2 in row[team_col][i+1:]
                ]),
                axis=1
            )
            feature_names.append(avg_feature)
        
        return X, feature_names
    
    def _calculate_hero_vs_hero_winrates(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate hero versus hero winrates."""
        logger.info("Calculating hero versus hero winrates...")
        feature_names = []
        
        team = 'radiant'
        enemy_team = 'dire'
        team_col = f'{team}_team'
        enemy_col = f'{enemy_team}_team'
        
        # Calculate team average counter winrate
        avg_feature = f'{team}_avg_vs_enemy_winrate'
        X[avg_feature] = X.apply(
            lambda row: np.mean([
                self.hero_stats['vs_hero_winrate'][hero][enemy]
                for hero in row[team_col]
                for enemy in row[enemy_col]
            ]),
            axis=1
        )
        feature_names.append(avg_feature)
        
        return X, feature_names
    
    def _calculate_hero_pickrates(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate hero pickrates within team."""
        logger.info("Calculating hero pickrates within team...")
        feature_names = []
        
        for team in ['radiant', 'dire']:
            team_col = f'{team}_team'
            
            # Calculate team average pickrate
            avg_feature = f'{team}_avg_hero_pickrate'
            X[avg_feature] = X.apply(
                lambda row: np.mean([
                    self.hero_stats['pickrate'][hero_id]
                    for hero_id in row[team_col]
                ]),
                axis=1
            )
            feature_names.append(avg_feature)
        
        return X, feature_names
    
    # def _calculate_hero_banrates(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    #     """Calculate hero banrates within team."""
    #     logger.info("Calculating hero banrates within team...")
    #     feature_names = []
        
    #     for team in ['radiant', 'dire']:
    #         team_heroes = [f'{team}_hero_{i}' for i in range(5)]
            
    #         # Calculate individual hero banrates
    #         for hero_col in team_heroes:
    #             feature_name = f'{team}_{hero_col}_banrate'
    #             X[feature_name] = X[hero_col].map(self.hero_stats['banrate'])
    #             feature_names.append(feature_name)
            
    #         # Calculate team average banrate
    #         avg_feature = f'{team}_avg_hero_banrate'
    #         X[avg_feature] = X[[f'{team}_{col}_banrate' for col in team_heroes]].mean(axis=1)
    #         feature_names.append(avg_feature)
        
    #     return X, feature_names
    
    def fit(self, X: pd.DataFrame, y=None) -> 'HeroStatsTransformer':
        """Fit the transformer by loading hero data and calculating statistics."""
        self._load_heroes_data()
        self._calculate_hero_stats(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by adding hero statistics features."""
        if self.hero_stats is None:
            raise ValueError("Transformer must be fitted before transform")
        
        logger.info("Starting feature transformation...")
        
        # Calculate all feature sets
        X, winrate_features = self._calculate_hero_winrates(X)
        X, synergy_features = self._calculate_hero_with_hero_winrates(X)
        X, counter_features = self._calculate_hero_vs_hero_winrates(X)
        X, pickrate_features = self._calculate_hero_pickrates(X)
        # X, banrate_features = self._calculate_hero_banrates(X)
        
        # Combine all feature names
        self.feature_names = (
            winrate_features + 
            synergy_features + 
            counter_features + 
            pickrate_features
            # banrate_features
        )
        
        logger.info(f"Added {len(self.feature_names)} hero statistics features")
        ret_features = set(self.get_feature_names()).difference(self.exclude_features)
        return X[list(ret_features)]
    
    def get_feature_names(self, categorical: bool = False) -> List[str]:
        """Get the list of feature names added by this transformer."""
        if categorical:
            return []
        return self.feature_names 