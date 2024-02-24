from dataclasses import dataclass
from typing import Optional, List

"""
Maximum rank for the matches. 
Ranks are represented by integers 
(10-15: Herald, 
20-25: Guardian, 
30-35: Crusader, 
40-45: Archon, 
50-55: Legend, 
60-65: Ancient, 
70-75: Divine, 
80-85: Immortal). 
Each increment represents an additional star.
"""

RANKED_LOBBY_TYPE = 7
RANKED_GAME_MODE= 22

@dataclass
class PublicMatchData:
    match_id: Optional[int]
    match_seq_num: Optional[int] # hz
    radiant_win: Optional[bool]
    start_time: Optional[int]
    duration: Optional[int]
    lobby_type: Optional[int] # 7 - ranked
    game_mode: Optional[int] # 22 - ranked
    avg_rank_tier: Optional[int]
    num_rank_tier: Optional[int] # hz
    cluster: Optional[int] # hz
    radiant_team: Optional[List[int]] # hero ids
    dire_team: Optional[List[int]] # hero ids
