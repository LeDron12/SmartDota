import json
import logging
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from ..core.base import BaseTransformer

logger = logging.getLogger(__name__)

class PlayerStatsTransformer(BaseTransformer):
    """Transformer for player statistics features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.player_stats: Dict[int, Dict[str, float]] = {}  # steam_id -> stats
        self.player_matches: Dict[int, int] = {}  # steam_id -> matches played
        self.player_last_match: Dict[int, int] = {}  # steam_id -> last match timestamp
        self._is_fitted = False
        self.use_diff = config.get('use_diff', False)
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Calculate player statistics from the training data.
        
        Args:
            df: Training DataFrame
        """
        logger.info("Calculating player statistics from match data...")
        
        # Initialize statistics
        self.player_stats = {}
        self.player_matches = {}
        self.player_last_match = {}
        
        # Process each match
        for _, row in df.iterrows():
            match_data = json.loads(row.to_json())
            match_time = match_data.get('start_time', 0)
            
            # Process both teams
            for team in ['radiant', 'dire']:
                for pos in range(1, 6):
                    player_key = f"{team}_{pos}"
                    if player_key in match_data:
                        player = match_data[player_key]
                        steam_id = player.get('steam_id')
                        if steam_id:
                            # Initialize player stats if not exists
                            if steam_id not in self.player_stats:
                                self.player_stats[steam_id] = {
                                    'kills': 0, 'deaths': 0, 'assists': 0,
                                    'last_hits': 0, 'denies': 0,
                                    'hero_damage': 0, 'hero_healing': 0,
                                    'role_match': 0, 'behavior': 0,
                                    'account_level': 0, 'dota_plus': 0,
                                    'smurf_flag': 0
                                }
                                self.player_matches[steam_id] = 0
                                self.player_last_match[steam_id] = 0
                            
                            # Update statistics
                            stats = self.player_stats[steam_id]
                            stats['kills'] += player.get('kills') or 0
                            stats['deaths'] += player.get('deaths') or 0
                            stats['assists'] += player.get('assists') or 0
                            stats['last_hits'] += player.get('numLastHits') or 0
                            stats['denies'] += player.get('numDenies') or 0
                            stats['hero_damage'] += player.get('heroDamage') or 0
                            stats['hero_healing'] += player.get('heroHealing') or 0
                            stats['role_match'] += int(player.get('role') == player.get('roleBasic')) 
                            stats['behavior'] += player.get('behavior') or 0
                            stats['account_level'] += player.get('dotaAccountLevel') or 0
                            stats['dota_plus'] += int(bool(player.get('isDotaPlusSubscriber')))
                            stats['smurf_flag'] += player.get('smurfFlag') or 0
                            
                            # Update match counts and last match time
                            self.player_matches[steam_id] += 1
                            self.player_last_match[steam_id] = max(
                                self.player_last_match[steam_id],
                                match_time
                            )
        
        # Calculate averages
        for steam_id in self.player_stats:
            matches = self.player_matches[steam_id]
            if matches > 0:
                for stat in self.player_stats[steam_id]:
                    self.player_stats[steam_id][stat] /= matches
        
        self._is_fitted = True
        logger.info("Finished calculating player statistics")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using calculated player statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        logger.info("Starting feature transformation...")
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(index=df.index)
        
        # Process each match
        for idx, row in df.iterrows():
            match_data = json.loads(row.to_json())
            match_time = match_data.get('start_time', 0)
            
            # Initialize team statistics
            team_stats = {
                'radiant': {
                    'kills': [], 'deaths': [], 'assists': [],
                    'last_hits': [], 'denies': [],
                    'hero_damage': [], 'hero_healing': [],
                    'role_match': [], 'behavior': [],
                    'account_level': [], 'dota_plus': [],
                    'smurf_flag': [], 'time_since_last': [],
                    'matches_played': []
                },
                'dire': {
                    'kills': [], 'deaths': [], 'assists': [],
                    'last_hits': [], 'denies': [],
                    'hero_damage': [], 'hero_healing': [],
                    'role_match': [], 'behavior': [],
                    'account_level': [], 'dota_plus': [],
                    'smurf_flag': [], 'time_since_last': [],
                    'matches_played': []
                }
            }
            
            # First pass: collect stats for players that exist in our database
            for team in ['radiant', 'dire']:
                for pos in range(1, 6):
                    player_key = f"{team}_{pos}"
                    if player_key in match_data:
                        player = match_data[player_key]
                        steam_id = player.get('steam_id')
                        
                        if steam_id and steam_id in self.player_stats:
                            stats = self.player_stats[steam_id]
                            for stat in stats:
                                team_stats[team][stat].append(stats[stat])
                            
                            # Add time since last match
                            time_since = match_time - self.player_last_match[steam_id]
                            team_stats[team]['time_since_last'].append(time_since)
                            
                            # Add matches played
                            team_stats[team]['matches_played'].append(
                                self.player_matches[steam_id]
                            )
            
            # Second pass: fill missing players with team averages
            for team in ['radiant', 'dire']:
                for pos in range(1, 6):
                    player_key = f"{team}_{pos}"
                    if player_key in match_data:
                        player = match_data[player_key]
                        steam_id = player.get('steam_id')
                        
                        if not steam_id or steam_id not in self.player_stats:
                            # Calculate team averages for each stat
                            for stat in team_stats[team]:
                                values = team_stats[team][stat]
                                if values:  # If we have any values from teammates
                                    team_stats[team][stat].append(np.mean(values))
                                else:  # If no teammates have stats, use 0
                                    team_stats[team][stat].append(0)
            
            # Calculate team averages
            team_avgs = {}
            for team in ['radiant', 'dire']:
                team_avgs[team] = {}
                for stat in team_stats[team]:
                    values = team_stats[team][stat]
                    if values:
                        team_avgs[team][stat] = np.mean(values)
                    else:
                        team_avgs[team][stat] = 0
            
            if self.use_diff:
                # Only add difference features
                for stat in team_stats['radiant']:
                    diff = team_avgs['radiant'][stat] - team_avgs['dire'][stat]
                    result_df.loc[idx, f'diff_{stat}'] = diff
            else:
                # Add team average features
                for team in ['radiant', 'dire']:
                    for stat in team_stats[team]:
                        result_df.loc[idx, f'{team}_avg_{stat}'] = team_avgs[team][stat]
        
        logger.info(f"Added {len(result_df.columns)} player statistics features")
        return result_df
    
    def get_feature_names(self, categorical: bool = False) -> List[str]:
        """
        Get the names of all features added by this transformer.
        
        Returns:
            List of feature names
        """
        if categorical:
            return []

        features = []
        stats = [
            'kills', 'deaths', 'assists', 'last_hits', 'denies',
            'hero_damage', 'hero_healing', 'role_match', 'behavior',
            'account_level', 'dota_plus', 'smurf_flag',
            'time_since_last', 'matches_played'
        ]
        
        if self.use_diff:
            # Only add difference features
            for stat in stats:
                features.append(f'diff_{stat}')
        else:
            # Add team average features
            for team in ['radiant', 'dire']:
                for stat in stats:
                    features.append(f'{team}_avg_{stat}')
        
        return features