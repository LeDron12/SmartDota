# Data Collection Module

This module handles fetching and preprocessing Dota 2 match data from OpenDota and Stratz APIs.

## 📊 Data Sources

- **OpenDota API**: Primary source for match data, player statistics, and hero information
- **Stratz API**: Secondary source for additional match statistics and player ratings

## 🛠️ Usage

**Run all scripts from CURRENT FOLDER**
```bash
cd src/data_new/
```

### Fetching Match Data

```python
from src.data_new.fetch_matches import fetch_opendota_matches

# Fetch match data
match_data, _ = fetch_opendota_matches('match_data', {'match_id': 1234567890})
```

### Fetching Player Statistics

```python
from src.data_new.fetch_stratz_matches import get_match_details, RateLimiter

# Initialize rate limiter and load API key
rate_limiter = RateLimiter()
with open('path/to/api_key.txt', 'r') as f:
    api_key = f.read().strip()

# Get detailed match information including player stats
match_details, status_code = get_match_details(
    match_id=1234567890,
    api_key=api_key,
    rate_limiter=rate_limiter,
    query_name='get_match_by_id'  # Name of your GraphQL query file
)
```

## 📁 Data Structure

### Match Data Fields

- `match_id`: Unique identifier for the match
- `radiant_win`: Boolean indicating if Radiant team won
- `duration`: Match duration in seconds
- `game_mode`: Game mode identifier
- `skill_level`: Approximate skill bracket
- `players`: List of player data including:
  - `account_id`: Player's Steam ID
  - `hero_id`: Hero identifier
  - `kills`, `deaths`, `assists`: Player performance metrics
  - `gold_per_min`, `xp_per_min`: Economic metrics
  - `hero_damage`, `hero_healing`: Combat metrics

### Rate Limiting

Both APIs implement rate limiting:
- OpenDota: 60 requests per minute
- Stratz: 100 requests per minute

The module includes built-in rate limiting to prevent API throttling.

## 🔧 Configuration

You can add you own scripts to 
1. sql/ if using OpenDota API (Paid):
> Docs: https://docs.opendota.com \
> Get API key: https://www.opendota.com/api-keys [No key needed for few requests]
2. graphQL/ if using Starz API (Free):
> Docs: https://api.stratz.com/graphiql \
> Get API key: https://stratz.com/api [Essential. Put to keys/]

## 📝 Notes

- Match data is cached locally to reduce API calls
- Historical data is available for matches up to years old
- Some fields may be missing for matches

## 🚀 Example

```python
from src.data_new.fetch_matches import fetch_opendota_matches
from src.data_new.fetch_stratz_matches import get_match_details

# Fetch basic match data
match_data, _ = fetch_opendota_matches('match_data', {'match_id': 1234567890})

# Get detailed player statistics
match_details = get_match_details(match_id=1234567890)

# Process and analyze the data
print(f"Match duration: {match_data['duration']} seconds")
print(f"Winner: {'Radiant' if match_data['radiant_win'] else 'Dire'}")
``` 