from typing import Dict, Any, Type, List
import pandas as pd
from pathlib import Path
import logging
import json
import random
import copy
# from src.data.api.OpenDota.public_matches_dataloader import PublicMatchesDataloader
# from src.data.dataclasses.public_match import PublicMatchData

logger = logging.getLogger(__name__)

class DatasetFactory:
    """Factory class for creating and loading datasets."""
    
    def __init__(self):
        """Initialize the dataset factory."""
        self._loaders = {
            'public_matches': self._load_public_matches,
            'starz_public_matches': self._load_starz_public_matches
        }
        self.no_role_cnt = {"matches": 0, "players": 0}
    
    def load_dataset(self, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
        """
        Load dataset based on configuration.
        
        Args:
            config: Dataset configuration dictionary
            project_root: Path to project root
            
        Returns:
            Raw dataset as DataFrame
        """
        dataset_type = config.get('type')
        if dataset_type not in self._loaders:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        logger.info(f"Loading dataset of type: {dataset_type}")
        return self._loaders[dataset_type](config, project_root)
    
    def _load_public_matches(self, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
        """
        Load public matches dataset.
        
        Args:
            config: Dataset configuration dictionary
            project_root: Path to project root
            
        Returns:
            DataFrame containing raw match data
        """
        data_path = project_root / config['path']
        params = config.get('params', {})
        
        logger.info(f"Loading public matches from: {data_path}")

        # Load JSON data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
        matches = data.get('rows', [])

        # Check for non-None and non-empty teams
        original_count = len(matches)
        matches = [match for match in matches if match.get('radiant_team') and match.get('dire_team')]
        filtered_count = len(matches)
        logger.info(f"Removed {original_count - filtered_count} matches due to missing or empty team data.")

        df = pd.DataFrame([
            {
                'match_id': match['match_id'],
                'start_time': match['start_time'],
                'radiant_team': match['radiant_team'],
                'dire_team': match['dire_team'],
                'radiant_win': match['radiant_win']
            }
            for match in matches
        ])

        df = df.sort_values(by='match_id', ascending=True)
        df = df.reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} matches")
        return df, df["radiant_win"].tolist()
    
    def _load_starz_public_matches(self, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:

        data_path = project_root / config['path']
        params = config.get('params', {})
        
        logger.info(f"Loading starz public matches from: {data_path}")

        # Load JSON data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
        matches = data.get('matches', [])

        # Check for non-None and non-empty teams
        original_count = len(matches)
        matches = list(filter(lambda x: 'public' not in  x, matches))
        filtered_count = len(matches)
        logger.info(f"Removed {original_count - filtered_count} matches due to missing or empty data.")

        df = self.convert_matches_to_dataframe(copy.deepcopy(matches))

        logger.info(f"Loaded {len(df)} matches")
        logger.info(f"No role count: {self.no_role_cnt}")

        df = df.sort_values(by='match_id', ascending=True)
        df = df.reset_index(drop=True)

        return df, df["radiant_win"].tolist()
    
    def flatten_steam_account(self, player_data):
        """Flatten the steamAccount dictionary."""
        steam_data = player_data.pop('steamAccount')
        for key, value in steam_data.items():
            player_data[f'{"steam_" if key == "id" else ""}{key}'] = value
        return player_data

    def get_team_position(self, player_data):
        """Get team and position for a player."""
        team = 'radiant' if player_data['isRadiant'] else 'dire'
        if not player_data['position']:
            return team, None
        position = player_data['position'].lower().replace('position_', '')
        return team, f"{team}_{position}"

    def convert_match_to_dataframe(self, match_data):
        """Convert a single match data structure to a pandas DataFrame."""
        # Extract base match info
        match_info = {
            'match_id': match_data['id'],
            'radiant_win': match_data['didRadiantWin'],
            'start_time': match_data['startDateTime'],
            'end_time': match_data['endDateTime'],
            'actualRank': match_data['actualRank'],
            'rank': match_data['rank'],
            'radiant_team': [player['heroId'] for player in match_data['players'] if player['isRadiant']],
            'dire_team': [player['heroId'] for player in match_data['players'] if not player['isRadiant']]
        }
        
        # Initialize player positions with None
        player_positions = {
            'radiant_1': None, 'radiant_2': None, 'radiant_3': None, 'radiant_4': None, 'radiant_5': None,
            'dire_1': None, 'dire_2': None, 'dire_3': None, 'dire_4': None, 'dire_5': None
        }
        
        # Fill in player data
        match_players = sorted(match_data['players'], key=lambda x: x['position'] is not None, reverse=True)
        if None in [player['position'] for player in match_players]:
            self.no_role_cnt["matches"] += 1

        for player in match_players:
            # Get team_position identifier
            player = self.flatten_steam_account(player)
            team, team_pos = self.get_team_position(player)
            
            # Store player data in the corresponding position
            if team_pos is None:
                team_keys = [k for k, v in player_positions.items() if k.startswith(team) and v is None]
                team_pos = random.choice(team_keys)
                self.no_role_cnt["players"] += 1
            player_positions[team_pos] = player
        
        # Combine match info with player data
        result = {**match_info, **player_positions}
        
        return pd.DataFrame([result])

    def convert_matches_to_dataframe(self, matches_data):
        """Convert a list of match data structures to a pandas DataFrame."""
        dfs = []
        for match in matches_data:
            if 'public' in match:
                continue
            
            df = self.convert_match_to_dataframe(match)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)