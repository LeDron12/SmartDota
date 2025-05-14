import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from ..core.base import BaseTransformer

logger = logging.getLogger(__name__)

class PlayerHeroStatsTransformer(BaseTransformer):
    """Transformer for player-hero statistics features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer.
        
        Args:
            config: Configuration dictionary with optional 'use_diff' flag
        """
        super().__init__(config)
        self.use_diff = config.get('use_diff', False)
        
        # Player personal stats
        self.player_winrates: Dict[int, float] = {}  # steam_id -> winrate
        self.player_matches: Dict[int, int] = {}  # steam_id -> matches played
        
        # Player-hero stats (for ally heroes)
        self.player_hero_winrates: Dict[Tuple[int, int], float] = {}  # (steam_id, hero_id) -> winrate
        self.player_hero_matches: Dict[Tuple[int, int], int] = {}  # (steam_id, hero_id) -> matches
        
        # Player vs hero stats (for enemy heroes)
        self.player_vs_hero_winrates: Dict[Tuple[int, int], float] = {}  # (steam_id, hero_id) -> winrate
        self.player_vs_hero_matches: Dict[Tuple[int, int], int] = {}  # (steam_id, hero_id) -> matches
        
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Calculate player-hero statistics from the training data.
        
        Args:
            df: Training DataFrame
        """
        logger.info("Calculating player-hero statistics from match data...")
        
        # Initialize statistics
        self.player_winrates = {}
        self.player_matches = {}
        self.player_hero_winrates = {}
        self.player_hero_matches = {}
        self.player_vs_hero_winrates = {}
        self.player_vs_hero_matches = {}
        
        # Process each match
        for _, row in df.iterrows():
            match_data = json.loads(row.to_json())
            radiant_win = match_data.get('radiant_win', False)
            
            # Process both teams
            for team in ['radiant', 'dire']:
                team_win = radiant_win if team == 'radiant' else not radiant_win
                
                # Get all heroes in this team
                team_heroes = []
                for pos in range(1, 6):
                    player_key = f"{team}_{pos}"
                    if player_key in match_data:
                        player = match_data[player_key]
                        steam_id = player.get('steam_id')
                        hero_id = player.get('heroId')
                        if steam_id and hero_id:
                            team_heroes.append((steam_id, hero_id))
                
                # Get all heroes in enemy team
                enemy_team = 'dire' if team == 'radiant' else 'radiant'
                enemy_heroes = []
                for pos in range(1, 6):
                    player_key = f"{enemy_team}_{pos}"
                    if player_key in match_data:
                        player = match_data[player_key]
                        hero_id = player.get('heroId')
                        if hero_id:
                            enemy_heroes.append(hero_id)
                
                # Update statistics for each player
                for steam_id, hero_id in team_heroes:
                    # Update player personal stats
                    if steam_id not in self.player_winrates:
                        self.player_winrates[steam_id] = 0
                        self.player_matches[steam_id] = 0
                    
                    self.player_winrates[steam_id] += int(team_win)
                    self.player_matches[steam_id] += 1
                    
                    # Update player-hero stats (for ally heroes)
                    for ally_steam_id, ally_hero_id in team_heroes:
                        key = (steam_id, ally_hero_id)
                        if key not in self.player_hero_winrates:
                            self.player_hero_winrates[key] = 0
                            self.player_hero_matches[key] = 0
                        
                        self.player_hero_winrates[key] += int(team_win)
                        self.player_hero_matches[key] += 1
                    
                    # Update player vs hero stats (for enemy heroes)
                    for enemy_hero_id in enemy_heroes:
                        key = (steam_id, enemy_hero_id)
                        if key not in self.player_vs_hero_winrates:
                            self.player_vs_hero_winrates[key] = 0
                            self.player_vs_hero_matches[key] = 0
                        
                        self.player_vs_hero_winrates[key] += int(team_win)
                        self.player_vs_hero_matches[key] += 1
        
        # Calculate averages
        for steam_id in self.player_winrates:
            matches = self.player_matches[steam_id]
            if matches > 0:
                self.player_winrates[steam_id] /= matches
        
        for key in self.player_hero_winrates:
            matches = self.player_hero_matches[key]
            if matches > 0:
                self.player_hero_winrates[key] /= matches
        
        for key in self.player_vs_hero_winrates:
            matches = self.player_vs_hero_matches[key]
            if matches > 0:
                self.player_vs_hero_winrates[key] /= matches
        
        self._is_fitted = True
        logger.info("Finished calculating player-hero statistics")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using calculated player-hero statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame with either team-specific features or difference features
            depending on use_diff flag
        """
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        logger.info("Starting feature transformation...")
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(index=df.index)
        
        # Process each match
        for idx, row in df.iterrows():
            match_data = json.loads(row.to_json())
            
            # Initialize team statistics
            team_stats = {
                'radiant': {
                    'player_winrate': [],
                    'player_matches': [],
                    'player_hero_winrate': [],
                    'player_hero_matches': [],
                    'player_vs_hero_winrate': [],
                    'player_vs_hero_matches': []
                },
                'dire': {
                    'player_winrate': [],
                    'player_matches': [],
                    'player_hero_winrate': [],
                    'player_hero_matches': [],
                    'player_vs_hero_winrate': [],
                    'player_vs_hero_matches': []
                }
            }
            
            # First pass: collect stats for players that exist in our database
            for team in ['radiant', 'dire']:
                # Get all heroes in this team
                team_heroes = []
                for pos in range(1, 6):
                    player_key = f"{team}_{pos}"
                    if player_key in match_data:
                        player = match_data[player_key]
                        steam_id = player.get('steam_id')
                        hero_id = player.get('heroId')
                        if steam_id and hero_id:
                            team_heroes.append((steam_id, hero_id))
                
                # Get all heroes in enemy team
                enemy_team = 'dire' if team == 'radiant' else 'radiant'
                enemy_heroes = []
                for pos in range(1, 6):
                    player_key = f"{enemy_team}_{pos}"
                    if player_key in match_data:
                        player = match_data[player_key]
                        hero_id = player.get('heroId')
                        if hero_id:
                            enemy_heroes.append(hero_id)
                
                # Process each player in the team
                for steam_id, hero_id in team_heroes:
                    # Personal stats
                    if steam_id in self.player_winrates:
                        team_stats[team]['player_winrate'].append(self.player_winrates[steam_id])
                        team_stats[team]['player_matches'].append(self.player_matches[steam_id])
                    
                    # Player-hero stats (for ally heroes)
                    hero_winrates = []
                    hero_matches = []
                    for ally_steam_id, ally_hero_id in team_heroes:
                        key = (steam_id, ally_hero_id)
                        if key in self.player_hero_winrates:
                            hero_winrates.append(self.player_hero_winrates[key])
                            hero_matches.append(self.player_hero_matches[key])
                    
                    if hero_winrates:
                        team_stats[team]['player_hero_winrate'].append(np.mean(hero_winrates))
                        team_stats[team]['player_hero_matches'].append(np.mean(hero_matches))
                    
                    # Player vs hero stats (for enemy heroes)
                    vs_hero_winrates = []
                    vs_hero_matches = []
                    for enemy_hero_id in enemy_heroes:
                        key = (steam_id, enemy_hero_id)
                        if key in self.player_vs_hero_winrates:
                            vs_hero_winrates.append(self.player_vs_hero_winrates[key])
                            vs_hero_matches.append(self.player_vs_hero_matches[key])
                    
                    if vs_hero_winrates:
                        team_stats[team]['player_vs_hero_winrate'].append(np.mean(vs_hero_winrates))
                        team_stats[team]['player_vs_hero_matches'].append(np.mean(vs_hero_matches))
            
            # Second pass: fill missing players with team averages
            for team in ['radiant', 'dire']:
                for stat in team_stats[team]:
                    values = team_stats[team][stat]
                    if values:  # If we have any values from teammates
                        team_stats[team][stat].append(np.mean(values))
                    else:  # If no teammates have stats, use 0
                        team_stats[team][stat].append(0)
            
            # Calculate team averages and store in temporary dictionary
            temp_stats = {}
            for team in ['radiant', 'dire']:
                for stat in team_stats[team]:
                    values = team_stats[team][stat]
                    if values:
                        temp_stats[f'{team}_avg_{stat}'] = np.mean(values)
                    else:
                        temp_stats[f'{team}_avg_{stat}'] = 0
            
            # Add either team-specific features or difference features to result_df
            if self.use_diff:
                # Only add difference features
                for stat in team_stats['radiant'].keys():
                    radiant_value = temp_stats[f'radiant_avg_{stat}']
                    dire_value = temp_stats[f'dire_avg_{stat}']
                    result_df.loc[idx, f'diff_{stat}'] = radiant_value - dire_value
            else:
                # Only add team-specific features
                for team in ['radiant', 'dire']:
                    for stat in team_stats[team]:
                        result_df.loc[idx, f'{team}_avg_{stat}'] = temp_stats[f'{team}_avg_{stat}']
        
        logger.info(f"Added {len(result_df.columns)} player-hero statistics features")
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
            'player_winrate', 'player_matches',
            'player_hero_winrate', 'player_hero_matches',
            'player_vs_hero_winrate', 'player_vs_hero_matches'
        ]
        
        if self.use_diff:
            # Only return difference features
            for stat in stats:
                features.append(f'diff_{stat}')
        else:
            # Only return team-specific features
            for team in ['radiant', 'dire']:
                for stat in stats:
                    features.append(f'{team}_avg_{stat}')
        
        return features