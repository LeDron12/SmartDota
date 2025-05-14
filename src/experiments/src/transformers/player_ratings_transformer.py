import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from ..core.base import BaseTransformer

logger = logging.getLogger(__name__)

class PlayerRatingsTransformer(BaseTransformer):
    """Transformer for calculating player ELO and Glicko ratings."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer.
        
        Args:
            config: Configuration dictionary with optional flags:
                - use_elo: Whether to calculate ELO ratings (default: True)
                - use_glicko: Whether to calculate Glicko ratings (default: True)
                - rating_params: Dictionary of rating system parameters:
                    - elo_k: ELO K-factor (default: 16)
                    - glicko_rd: Initial Glicko RD (default: 350)
                    - glicko_vol: Initial Glicko volatility (default: 0.06)
                    - glicko_tau: Glicko system constant (default: 0.5)
                    - rating_decay: Decay factor for ratings over time (default: 0.95)
                - nan_handling: Dictionary of NaN handling parameters:
                    - default_rating: Default rating for NaN values (default: 1500.0)
                    - default_rd: Default RD for NaN values (default: 350.0)
                    - fill_method: How to fill NaN values ('default' or 'team_avg', default: 'team_avg')
                - bounds: Dictionary of rating bounds:
                    - min_rating: Minimum allowed rating (default: 500.0)
                    - max_rating: Maximum allowed rating (default: 3000.0)
                    - min_rd: Minimum allowed RD (default: 30.0)
                    - max_rd: Maximum allowed RD (default: 350.0)
        """
        super().__init__(config)
        self.use_elo = config.get('use_elo', True)
        self.use_glicko = config.get('use_glicko', True)
        self.add_diff = config.get('add_diff', True)

        # Rating system parameters
        rating_params = config.get('rating_params', {})
        self.elo_k = rating_params.get('elo_k', 16)
        self.glicko_rd = rating_params.get('glicko_rd', 350)
        self.glicko_vol = rating_params.get('glicko_vol', 0.06)
        self.glicko_tau = rating_params.get('glicko_tau', 0.5)
        self.rating_decay = rating_params.get('rating_decay', 0.95)
        
        # NaN handling parameters
        nan_params = config.get('nan_handling', {})
        self.default_rating = nan_params.get('default_rating', 1500.0)
        self.default_rd = nan_params.get('default_rd', 350.0)
        self.fill_method = nan_params.get('fill_method', 'team_avg')
        
        # Rating bounds
        bounds = config.get('bounds', {})
        self.min_rating = bounds.get('min_rating', 500.0)
        self.max_rating = bounds.get('max_rating', 3000.0)
        self.min_rd = bounds.get('min_rd', 30.0)
        self.max_rd = bounds.get('max_rd', 350.0)
        
        # Rating system states
        self.player_ratings: Dict[int, Dict[str, Dict[str, float]]] = {}  # steam_id -> rating_system -> {rating, rd, vol}
        self.player_last_match: Dict[int, int] = {}  # steam_id -> last match timestamp
        self._is_fitted = False
        
        logger.info(f"Initialized rating systems: ELO={self.use_elo}, Glicko={self.use_glicko}")
        logger.info(f"Rating parameters: K={self.elo_k}, decay={self.rating_decay}")
        logger.info(f"Rating bounds: [{self.min_rating}, {self.max_rating}] for ratings, [{self.min_rd}, {self.max_rd}] for RD")
    
    def _safe_mean(self, values: List[float]) -> float:
        """
        Calculate mean of values, handling NaN values.
        
        Args:
            values: List of values to average
            
        Returns:
            Mean value, using default if all values are NaN
        """
        # Filter out NaN values
        valid_values = [v for v in values if not pd.isna(v)]
        if not valid_values:
            return self.default_rating
        return np.mean(valid_values)
    
    def _fill_team_ratings(self, team_ratings: Dict[str, List[float]], 
                          rating_type: str) -> float:
        """
        Fill missing ratings for a team using appropriate method.
        
        Args:
            team_ratings: Dictionary of team ratings
            rating_type: Type of rating to fill
            
        Returns:
            Filled rating value
        """
        values = team_ratings[rating_type]
        if not values:
            # No values at all, use default
            return self.default_rd if rating_type == 'glicko_rd' else self.default_rating
            
        if self.fill_method == 'team_avg':
            # Use team average if available, otherwise default
            return self._safe_mean(values)
        else:
            # Use default value
            return self.default_rd if rating_type == 'glicko_rd' else self.default_rating
    
    def _validate_and_fill_ratings(self, ratings: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and fill NaN values in player ratings.
        
        Args:
            ratings: Dictionary of player ratings
            
        Returns:
            Dictionary with filled ratings
        """
        filled_ratings = {}
        for system, values in ratings.items():
            filled_ratings[system] = {}
            for key, value in values.items():
                if pd.isna(value):
                    if key == 'rd':
                        filled_ratings[system][key] = self.default_rd
                    else:
                        filled_ratings[system][key] = self.default_rating
                else:
                    filled_ratings[system][key] = float(value)
        return filled_ratings
    
    def _initialize_player_ratings(self, steam_id: int) -> None:
        """Initialize rating systems for a new player."""
        if steam_id not in self.player_ratings:
            self.player_ratings[steam_id] = {}
            if self.use_elo:
                self.player_ratings[steam_id]['elo'] = {'rating': 1500.0}
            if self.use_glicko:
                self.player_ratings[steam_id]['glicko'] = {
                    'rating': 1500.0,
                    'rd': self.glicko_rd,
                    'vol': self.glicko_vol
                }
    
    def _clip_rating(self, rating: float) -> float:
        """Clip rating to allowed bounds."""
        return np.clip(rating, self.min_rating, self.max_rating)
    
    def _clip_rd(self, rd: float) -> float:
        """Clip RD to allowed bounds."""
        return np.clip(rd, self.min_rd, self.max_rd)
    
    def _update_elo_rating(self, player_id: int, opponent_ids: List[int], 
                          player_team_won: bool) -> None:
        """Update ELO rating for a player."""
        if not self.use_elo:
            return
            
        self._initialize_player_ratings(player_id)
        player_rating = self.player_ratings[player_id]['elo']['rating']
        
        # Calculate average opponent rating
        opponent_ratings = []
        for opp_id in opponent_ids:
            if opp_id in self.player_ratings and 'elo' in self.player_ratings[opp_id]:
                opponent_ratings.append(self.player_ratings[opp_id]['elo']['rating'])
            else:
                opponent_ratings.append(self.default_rating)
        
        avg_opponent_rating = np.mean(opponent_ratings)
        
        # Calculate expected score using logistic function
        expected_score = 1 / (1 + 10 ** ((avg_opponent_rating - player_rating) / 400))
        actual_score = 1.0 if player_team_won else 0.0
        
        # Update rating with bounds
        rating_change = self.elo_k * (actual_score - expected_score)
        new_rating = self._clip_rating(player_rating + rating_change)
        self.player_ratings[player_id]['elo']['rating'] = new_rating
    
    def _update_glicko_rating(self, player_id: int, opponent_ids: List[int],
                             player_team_won: bool) -> None:
        """Update Glicko rating for a player."""
        if not self.use_glicko:
            return
            
        self._initialize_player_ratings(player_id)
        player_state = self.player_ratings[player_id]['glicko']
        mu = (player_state['rating'] - 1500) / 173.7178
        phi = player_state['rd'] / 173.7178
        sigma = player_state['vol']
        
        # Calculate average opponent rating and RD
        opponent_mus = []
        opponent_phis = []
        for opp_id in opponent_ids:
            if opp_id in self.player_ratings and 'glicko' in self.player_ratings[opp_id]:
                opp_state = self.player_ratings[opp_id]['glicko']
                opponent_mus.append((opp_state['rating'] - 1500) / 173.7178)
                opponent_phis.append(opp_state['rd'] / 173.7178)
            else:
                opponent_mus.append(0)
                opponent_phis.append(self.glicko_rd / 173.7178)
        
        # Calculate v (variance) and delta (rating change)
        v = 0
        delta = 0
        actual_score = 1.0 if player_team_won else 0.0
        
        for opp_mu, opp_phi in zip(opponent_mus, opponent_phis):
            g = 1 / np.sqrt(1 + 3 * opp_phi**2 / np.pi**2)
            E = 1 / (1 + np.exp(-g * (mu - opp_mu)))
            v += g**2 * E * (1 - E)
            delta += g * (actual_score - E)
        
        if v == 0:  # Handle case with no valid opponents
            return
            
        v = 1 / v
        delta = v * delta
        
        # Update volatility using iterative algorithm
        a = np.log(sigma**2)
        A = a
        B = 0
        if delta**2 > phi**2 + v:
            B = np.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while self._glicko_f(a - k * self.glicko_tau, delta, phi, v, a) < 0:
                k += 1
            B = a - k * self.glicko_tau
        
        fA = self._glicko_f(A, delta, phi, v, a)
        fB = self._glicko_f(B, delta, phi, v, a)
        
        for _ in range(20):  # Maximum 20 iterations
            C = A + (A - B) * fA / (fB - fA)
            fC = self._glicko_f(C, delta, phi, v, a)
            if abs(fC) < 1e-6:
                break
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2
            B = C
            fB = fC
        
        sigma_new = np.exp(A / 2)
        
        # Update rating and RD with bounds
        phi_star = np.sqrt(phi**2 + sigma_new**2)
        phi_new = 1 / np.sqrt(1 / phi_star**2 + 1 / v)
        mu_new = mu + phi_new**2 * delta / v
        
        # Convert back to Glicko scale and apply bounds
        new_rating = self._clip_rating(173.7178 * mu_new + 1500)
        new_rd = self._clip_rd(173.7178 * phi_new)
        
        self.player_ratings[player_id]['glicko'].update({
            'rating': new_rating,
            'rd': new_rd,
            'vol': sigma_new
        })
    
    def _glicko_f(self, x: float, delta: float, phi: float, v: float, a: float) -> float:
        """Helper function for Glicko-2 rating system."""
        ex = np.exp(x)
        return (ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex)**2) - 
                (x - a) / self.glicko_tau**2)
    
    def _apply_time_decay(self, player_id: int, current_time: int) -> None:
        """Apply time decay to player ratings."""
        if player_id not in self.player_last_match:
            return
            
        time_diff = current_time - self.player_last_match[player_id]
        if time_diff <= 0:
            return
            
        # Apply decay based on time difference (in hours)
        hours_diff = time_diff / 3600
        decay_factor = self.rating_decay ** hours_diff
        
        if self.use_elo and 'elo' in self.player_ratings[player_id]:
            current_rating = self.player_ratings[player_id]['elo']['rating']
            decayed_rating = self.default_rating + (current_rating - self.default_rating) * decay_factor
            self.player_ratings[player_id]['elo']['rating'] = self._clip_rating(decayed_rating)
            
        if self.use_glicko and 'glicko' in self.player_ratings[player_id]:
            current_rating = self.player_ratings[player_id]['glicko']['rating']
            current_rd = self.player_ratings[player_id]['glicko']['rd']
            
            # Decay rating towards default
            decayed_rating = self.default_rating + (current_rating - self.default_rating) * decay_factor
            self.player_ratings[player_id]['glicko']['rating'] = self._clip_rating(decayed_rating)
            
            # Increase RD over time (more uncertainty)
            rd_increase = min(self.max_rd - current_rd, (1 - decay_factor) * self.glicko_rd)
            self.player_ratings[player_id]['glicko']['rd'] = self._clip_rd(current_rd + rd_increase)
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Calculate player ratings from the training data.
        
        Args:
            df: Training DataFrame with match data
        """
        logger.info("Calculating player ratings from match data...")
        
        # Initialize ratings
        self.player_ratings = {}
        self.player_last_match = {}
        
        # Sort by match time to ensure chronological processing
        df = df.sort_values('start_time')
        
        # Process each match
        for _, row in df.iterrows():
            match_data = row.to_dict()
            match_time = int(match_data['start_time'])
            radiant_win = match_data.get('radiant_win', False)
            
            # Process both teams
            for team in ['radiant', 'dire']:
                team_players = []
                opponent_team = 'dire' if team == 'radiant' else 'radiant'
                
                # Collect team players
                for pos in range(1, 6):
                    player_key = f"{team}_{pos}"
                    if player_key in match_data:
                        steam_id = match_data[player_key].get('steam_id')
                        if steam_id and not pd.isna(steam_id):
                            player_id = int(steam_id)
                            team_players.append(player_id)
                            
                            # Apply time decay if player has previous matches
                            self._apply_time_decay(player_id, match_time)
                            self.player_last_match[player_id] = match_time
                
                # Collect opponent players
                opponent_ids = []
                for pos in range(1, 6):
                    opp_key = f"{opponent_team}_{pos}"
                    if opp_key in match_data:
                        opp_id = match_data[opp_key].get('steam_id')
                        if opp_id and not pd.isna(opp_id):
                            opponent_id = int(opp_id)
                            opponent_ids.append(opponent_id)
                            
                            # Apply time decay if opponent has previous matches
                            self._apply_time_decay(opponent_id, match_time)
                            self.player_last_match[opponent_id] = match_time
                
                # Update ratings for each player
                player_team_won = (team == 'radiant') == radiant_win
                for player_id in team_players:
                    self._update_elo_rating(player_id, opponent_ids, player_team_won)
                    self._update_glicko_rating(player_id, opponent_ids, player_team_won)
        
        # Validate and fill any NaN values in ratings
        for steam_id in self.player_ratings:
            self.player_ratings[steam_id] = self._validate_and_fill_ratings(
                self.player_ratings[steam_id]
            )
        
        self._is_fitted = True
        logger.info("Finished calculating player ratings")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using calculated player ratings.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame with rating features, guaranteed to have no NaN values
            and all values within reasonable bounds
        """
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        logger.info("Starting rating feature transformation...")
        
        # Initialize result DataFrame with zeros to ensure no NaN values
        result_df = pd.DataFrame(0, index=df.index, columns=[
            f'{team}_avg_{rating_type}'
            for team in ['radiant', 'dire']
            for rating_type in (['elo_rating'] if self.use_elo else []) + 
                             (['glicko_rating', 'glicko_rd'] if self.use_glicko else [])
        ] + [
            f'diff_{rating_type}'
            for rating_type in (['elo_rating'] if self.use_elo else []) + 
                             (['glicko_rating', 'glicko_rd'] if self.use_glicko else [])
        ] if self.add_diff else [])
        
        # Sort by match time to ensure chronological processing
        df = df.sort_values('start_time')
        
        # Process each match
        for idx, row in df.iterrows():
            match_data = row.to_dict()
            match_time = int(match_data['start_time'])
            
            # Initialize team ratings with default values
            team_ratings = {
                'radiant': {
                    'elo_rating': [],
                    'glicko_rating': [],
                    'glicko_rd': []
                },
                'dire': {
                    'elo_rating': [],
                    'glicko_rating': [],
                    'glicko_rd': []
                }
            }
            
            # First pass: collect ratings for known players
            for team in ['radiant', 'dire']:
                for pos in range(1, 6):
                    player_key = f"{team}_{pos}"
                    if player_key in match_data:
                        steam_id = match_data[player_key].get('steam_id')
                        if steam_id and not pd.isna(steam_id):
                            player_id = int(steam_id)
                            
                            # Apply time decay if player has previous matches
                            if player_id in self.player_last_match:
                                self._apply_time_decay(player_id, match_time)
                            
                            if player_id in self.player_ratings:
                                ratings = self.player_ratings[player_id]
                                if self.use_elo and 'elo' in ratings:
                                    rating = float(ratings['elo']['rating'])
                                    if not pd.isna(rating):
                                        team_ratings[team]['elo_rating'].append(self._clip_rating(rating))
                                if self.use_glicko and 'glicko' in ratings:
                                    rating = float(ratings['glicko']['rating'])
                                    rd = float(ratings['glicko']['rd'])
                                    if not pd.isna(rating):
                                        team_ratings[team]['glicko_rating'].append(self._clip_rating(rating))
                                    if not pd.isna(rd):
                                        team_ratings[team]['glicko_rd'].append(self._clip_rd(rd))
            
            # Second pass: fill missing players with team averages or defaults
            for team in ['radiant', 'dire']:
                for rating_type in team_ratings[team]:
                    if not team_ratings[team][rating_type]:
                        # No valid ratings, use default
                        default_value = self._clip_rd(self.default_rd) if rating_type == 'glicko_rd' else self._clip_rating(self.default_rating)
                        team_ratings[team][rating_type] = [default_value]
            
            # Calculate team averages using safe mean
            team_avgs = {}
            for team in ['radiant', 'dire']:
                team_avgs[team] = {}
                for rating_type in team_ratings[team]:
                    values = team_ratings[team][rating_type]
                    if values:
                        avg = self._safe_mean(values)
                        if rating_type == 'glicko_rd':
                            team_avgs[team][rating_type] = self._clip_rd(float(avg))
                        else:
                            team_avgs[team][rating_type] = self._clip_rating(float(avg))
                    else:
                        if rating_type == 'glicko_rd':
                            team_avgs[team][rating_type] = self._clip_rd(self.default_rd)
                        else:
                            team_avgs[team][rating_type] = self._clip_rating(self.default_rating)
            
            # Add features with explicit float conversion and bounds
            for team in ['radiant', 'dire']:
                for rating_type in team_ratings[team]:
                    value = team_avgs[team][rating_type]
                    if rating_type == 'glicko_rd':
                        value = self._clip_rd(value)
                    else:
                        value = self._clip_rating(value)
                    result_df.loc[idx, f'{team}_avg_{rating_type}'] = float(value)
            
            # Add rating differences with explicit float conversion and bounds
            if self.add_diff:
                for rating_type in team_ratings['radiant']:
                    diff = float(team_avgs['radiant'][rating_type] - team_avgs['dire'][rating_type])
                    # Clip differences to reasonable range
                    max_diff = self.max_rating - self.min_rating
                    diff = np.clip(diff, -max_diff, max_diff)
                    result_df.loc[idx, f'diff_{rating_type}'] = diff
        
        # Verify no NaN values remain
        if result_df.isna().any().any():
            logger.warning("NaN values detected in output, filling with defaults")
            for col in result_df.columns:
                if 'glicko_rd' in col:
                    result_df[col] = result_df[col].fillna(self._clip_rd(self.default_rd))
                else:
                    result_df[col] = result_df[col].fillna(self._clip_rating(self.default_rating))
        
        # Final type conversion to ensure all values are float32
        result_df = result_df.astype(np.float32)
        
        logger.info(f"Added {len(result_df.columns)} rating features")
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
        rating_types = []
        if self.use_elo:
            rating_types.append('elo_rating')
        if self.use_glicko:
            rating_types.extend(['glicko_rating', 'glicko_rd'])
        
        # Add team average features
        for team in ['radiant', 'dire']:
            for rating_type in rating_types:
                features.append(f'{team}_avg_{rating_type}')
        
        if self.add_diff:
            # Add difference features
            for rating_type in rating_types:
                features.append(f'diff_{rating_type}')
        
        return features 