from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamData:
    tag: str
    name: str
    team_id: int
    rating: Optional[float]
    wins: Optional[int]
    losses: Optional[int]
    last_match_time: Optional[int] # Unix timestamp
