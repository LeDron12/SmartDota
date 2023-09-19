from dataclasses import dataclass

from typing import Dict, List, Union

@dataclass
class MatchData:
    match_id: int
    radiant_team_id: int
    dire_team_id: int
    picks: List[Dict[str, Union[int, str]]] # [{'order': 15, 'hero_id': 74, 'team_id': 'dire'}]
    patch_id: int

    def __post_init__(self) -> None:
        self.picks = sorted(self.picks, key=lambda x: x['order'])