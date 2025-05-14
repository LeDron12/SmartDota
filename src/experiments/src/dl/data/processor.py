import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from src.experiments.src.dl.config import DataConfig

@dataclass
class ProcessedMatch:
    match_id: int
    features: np.ndarray  # Shape: (game_length, n_features)
    label: int  # 1 for Radiant win, 0 for Dire win
    game_length: int

class MatchDataProcessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_scalers = {}  # Will store normalization parameters
        
    def _extract_teamfight_metrics(self, teamfight: Dict, is_radiant: bool) -> Dict:
        """Extract metrics for a specific team from a teamfight"""
        # First 5 players are Radiant, last 5 are Dire
        team_players = teamfight['players'][:5] if is_radiant else teamfight['players'][5:]
        
        # Calculate team-level metrics
        total_damage = sum(p.get('damage', 0) for p in team_players)
        total_healing = sum(p.get('healing', 0) for p in team_players)
        total_deaths = sum(p.get('deaths', 0) for p in team_players)
        total_gold_delta = sum(p.get('gold_delta', 0) for p in team_players)
        total_xp_delta = sum(p.get('xp_delta', 0) for p in team_players)
        
        # Calculate ability usage metrics
        ability_uses = {}
        for player in team_players:
            for ability, count in player.get('ability_uses', {}).items():
                ability_uses[ability] = ability_uses.get(ability, 0) + count
        
        # Calculate item usage metrics
        item_uses = {}
        for player in team_players:
            for item, count in player.get('item_uses', {}).items():
                item_uses[item] = item_uses.get(item, 0) + count
        
        return {
            'damage': total_damage,
            'healing': total_healing,
            'deaths': total_deaths,
            'gold_delta': total_gold_delta,
            'xp_delta': total_xp_delta,
            'ability_uses': ability_uses,
            'item_uses': item_uses
        }
    
    def _aggregate_teamfights(self, teamfights: List[Dict], game_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate teamfight data into per-minute metrics"""
        # Initialize arrays for each metric
        metrics = {
            'radiant_damage': np.zeros(game_length),
            'dire_damage': np.zeros(game_length),
            'radiant_healing': np.zeros(game_length),
            'dire_healing': np.zeros(game_length),
            'radiant_deaths': np.zeros(game_length),
            'dire_deaths': np.zeros(game_length),
            'radiant_gold_delta': np.zeros(game_length),
            'dire_gold_delta': np.zeros(game_length),
            'radiant_xp_delta': np.zeros(game_length),
            'dire_xp_delta': np.zeros(game_length),
            'teamfight_count': np.zeros(game_length)
        }
        
        for tf in teamfights:
            minute = int(tf['start'] / 60)
            if minute >= game_length:
                continue
                
            radiant_metrics = self._extract_teamfight_metrics(tf, True)
            dire_metrics = self._extract_teamfight_metrics(tf, False)
            
            # Update metrics for this minute
            metrics['radiant_damage'][minute] += radiant_metrics['damage']
            metrics['dire_damage'][minute] += dire_metrics['damage']
            metrics['radiant_healing'][minute] += radiant_metrics['healing']
            metrics['dire_healing'][minute] += dire_metrics['healing']
            metrics['radiant_deaths'][minute] += radiant_metrics['deaths']
            metrics['dire_deaths'][minute] += dire_metrics['deaths']
            metrics['radiant_gold_delta'][minute] += radiant_metrics['gold_delta']
            metrics['dire_gold_delta'][minute] += dire_metrics['gold_delta']
            metrics['radiant_xp_delta'][minute] += radiant_metrics['xp_delta']
            metrics['dire_xp_delta'][minute] += dire_metrics['xp_delta']
            metrics['teamfight_count'][minute] += 1
            
        # Stack all metrics into a single array
        stacked_metrics = np.stack([
            metrics['radiant_damage'], metrics['dire_damage'],
            metrics['radiant_healing'], metrics['dire_healing'],
            metrics['radiant_deaths'], metrics['dire_deaths'],
            metrics['radiant_gold_delta'], metrics['dire_gold_delta'],
            metrics['radiant_xp_delta'], metrics['dire_xp_delta'],
            metrics['teamfight_count']
        ])
        
        return stacked_metrics, metrics['radiant_damage'], metrics['dire_damage']
    
    def _process_tower_status(self, status: int, is_radiant: bool) -> int:
        """Convert tower status bitmask to count of remaining towers"""
        if is_radiant:
            return bin(status & 0b111111111111).count('1')
        return bin((status >> 12) & 0b111111111111).count('1')
    
    def _extract_objectives(self, match_data: Dict, game_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract objective-related features"""
        radiant_towers = np.zeros(game_length)
        dire_towers = np.zeros(game_length)
        first_blood = np.zeros(game_length)  # [radiant, dire]
        last_roshan_killed = np.zeros(game_length)  # [radiant, dire]
        
        # Initialize with full tower count (11 towers per team)
        initial_tower_count = 11
        radiant_towers[0] = initial_tower_count
        dire_towers[0] = initial_tower_count
        
        # Process objectives
        for obj in match_data.get('objectives', []):
            minute = int(obj['time'] / 60)
            if minute >= game_length:
                continue
                
            if obj['type'] == 'CHAT_MESSAGE_FIRSTBLOOD':
                if obj.get('player_slot', 0) < 128:
                    first_blood[minute:] = 1  # Radiant got first blood
                else:
                    first_blood[minute:] = -1  # Dire got first blood
            elif obj['type'] == 'building_kill':
                if 'tower' in obj['key']:
                    if 'badguys' in obj['key']:
                        # Radiant tower destroyed
                        radiant_towers[minute:] = max(0, radiant_towers[minute-1] - 1)
                    else:
                        # Dire tower destroyed
                        dire_towers[minute:] = max(0, dire_towers[minute-1] - 1)
            elif obj['type'] == 'CHAT_MESSAGE_ROSHAN_KILL':
                if obj.get('team', 0) == 2:  # Radiant team
                    last_roshan_killed[minute:] = 1
                else:  # Dire team
                    last_roshan_killed[minute:] = -1
        
        return radiant_towers, dire_towers, first_blood, last_roshan_killed
    
    def process_match(self, match_data: Dict) -> Optional[ProcessedMatch]:
        """Process a single match into features"""

        match_id = match_data['match_id']

        if not match_data.get('radiant_gold_adv'):
            return None
        gold_adv = np.array(match_data.get('radiant_gold_adv', []))[:self.config.max_game_length]
        gold_adv = np.pad(gold_adv, (0, self.config.max_game_length - len(gold_adv)))

        if not match_data.get('radiant_xp_adv'):
            return None
        xp_adv = np.array(match_data.get('radiant_xp_adv', []))[:self.config.max_game_length]
        xp_adv = np.pad(xp_adv, (0, self.config.max_game_length - len(xp_adv)))

        teamfight_metrics, _, _ = self._aggregate_teamfights(match_data.get('teamfights', []), self.config.max_game_length)

        radiant_towers, dire_towers, first_blood, roshan_kills = self._extract_objectives(
            match_data, self.config.max_game_length
        )
        
        final_features = [gold_adv, xp_adv]
        if "teamfight_count" in self.config.feature_columns:
            final_features.append(teamfight_metrics.T)
        if "radiant_towers" in self.config.feature_columns:
            final_features.append(radiant_towers)
            final_features.append(dire_towers)
            final_features.append(first_blood)
            final_features.append(roshan_kills)
        features = np.column_stack(final_features)
        
        ret = ProcessedMatch(
            match_id=match_id,
            features=features,
            label=1 if match_data.get('radiant_win', False) else 0,
            game_length=len(match_data.get('radiant_gold_adv', []))
        )
        return ret
    
    def process_matches(self, matches: List[Dict]) -> List[ProcessedMatch]:
        """Process multiple matches"""
        processed_matches = []
        for match in matches:
            processed = self.process_match(match)
            if processed is not None:
                processed_matches.append(processed)
        print('features.shape:', processed_matches[0].features.shape)
        return processed_matches
    
    # def save_scalers(self, path: str):
    #     """Save feature scalers to file"""
    #     with open(path, 'w') as f:
    #         json.dump(self.feature_scalers, f)
    
    # def load_scalers(self, path: str):
    #     """Load feature scalers from file"""
    #     with open(path, 'r') as f:
    #         self.feature_scalers = json.load(f) 