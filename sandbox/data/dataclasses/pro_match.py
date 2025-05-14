from dataclasses import dataclass
from typing import Optional


@dataclass
class ProMatchData:
    match_id: int
    duration: Optional[int]
    start_time: Optional[int]
    radiant_team_id: Optional[int]
    dire_team_id: Optional[int]
    leagueid: Optional[int]
    series_type: Optional[int]
    radiant_score: Optional[int]
    dire_score: Optional[int]
    radiant_win: Optional[bool]

@dataclass
class LiveProMatchData:
    match_id: int
    duration: Optional[int]
    start_time: Optional[int]
    radiant_team_id: Optional[int]
    dire_team_id: Optional[int]
    leagueid: Optional[int]
    series_type: Optional[int]
    radiant_score: Optional[int]
    dire_score: Optional[int]
    radiant_win: Optional[bool]
