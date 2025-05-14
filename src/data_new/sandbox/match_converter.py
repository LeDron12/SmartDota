import pandas as pd
import numpy as np

def flatten_steam_account(player_data):
    """Flatten the steamAccount dictionary."""
    steam_data = player_data.pop('steamAccount')
    for key, value in steam_data.items():
        player_data[f'{"steam_" if key == "id" else ""}{key}'] = value

def get_team_position(player_data):
    """Get team and position for a player."""
    team = 'radiant' if player_data['isRadiant'] else 'dire'
    position = player_data['position'].lower().replace('position_', '')
    return f"{team}_{position}"

def convert_match_to_dataframe(match_data):
    """Convert a single match data structure to a pandas DataFrame."""
    # Extract base match info
    match_info = {
        'match_id': match_data['id'],
        'did_radiant_win': match_data['didRadiantWin'],
        'start_time': match_data['startDateTime'],
        'end_time': match_data['endDateTime'],
        'rank': match_data['rank']
    }
    
    # Initialize player positions with None
    player_positions = {
        'radiant_1': None, 'radiant_2': None, 'radiant_3': None, 'radiant_4': None, 'radiant_5': None,
        'dire_1': None, 'dire_2': None, 'dire_3': None, 'dire_4': None, 'dire_5': None
    }
    
    # Fill in player data
    for player in match_data['players']:
        # Get team_position identifier
        player = flatten_steam_account(player)
        team_pos = get_team_position(player)
        
        # Store player data in the corresponding position
        player_positions[team_pos] = player
    
    # Combine match info with player data
    result = {**match_info, **player_positions}
    
    return pd.DataFrame([result])

def convert_matches_to_dataframe(matches_data):
    """Convert a list of match data structures to a pandas DataFrame."""
    dfs = []
    for match in matches_data:
        df = convert_match_to_dataframe(match)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

# Example usage:
if __name__ == "__main__":
    # Example match data
    example_match = [{
        'id': 8242057914,
        'didRadiantWin': False,
        'startDateTime': 1743800411,
        'endDateTime': 1743802995,
        'actualRank': 62,
        'rank': 62,
        'players': [{
            'playerSlot': 0,
            'isRadiant': True,
            'heroId': 103,
            'kills': 6,
            'deaths': 12,
            'assists': 12,
            'numLastHits': 95,
            'numDenies': 2,
            'heroDamage': 19621,
            'heroHealing': 0,
            'steamAccount': {
                'id': 88805322,
                'isDotaPlusSubscriber': True,
                'dotaAccountLevel': 160,
                'smurfFlag': 0
            },
            'lane': 'OFF_LANE',
            'position': 'POSITION_4',
            'role': 'LIGHT_SUPPORT',
            'roleBasic': 'LIGHT_SUPPORT',
            'behavior': 0
        }]
    }]
    
    # Convert to DataFrame
    df = convert_matches_to_dataframe(example_match)
    print("Columns:", df.columns.tolist())
    print("\nFirst row:")
    print(df.iloc[0]) 