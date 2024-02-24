from dataclasses import dataclass
from typing import Dict, List, Union, Optional

from src.data.dataclasses.draft import DraftItem, PickBanItem
from src.data.dataclasses.objective import ObjectiveData
from src.data.dataclasses.teamfight import TeamfightsData
from src.data.dataclasses.team import TeamData
from src.data.dataclasses.player import InGamePlayerData
from src.data.dataclasses.pro_match import ProMatchData
from src.data.dataclasses.public_match import PublicMatchData


# @dataclass
# class MatchData:
#     match_id: int
#     radiant_team_id: int
#     dire_team_id: int
#     picks: List[Dict[str, Union[int, str]]] # [{'order': 15, 'hero_id': 74, 'team_id': 'dire'}]
#     patch_id: int

#     def __post_init__(self) -> None:
#         self.picks = sorted(self.picks, key=lambda x: x['order'])

@dataclass
class MatchData:
    match_id: int
    public_match_data: Optional[PublicMatchData]
    pro_match_data: Optional[ProMatchData]
    barracks_status_dire: Optional[int] # хз че за циферка
    barracks_status_radiant: Optional[int] # хз че за циферка
    dire_score: Optional[int]
    draft_timings: Optional[List[DraftItem]]
    picks_bans: Optional[List[PickBanItem]]
    duration: Optional[int]
    first_blood_time: Optional[int]
    leagueid: Optional[int] # no train
    objectives: Optional[ObjectiveData]
    radiant_gold_adv: Optional[List[int]]
    radiant_score: Optional[int] 
    radiant_win: Optional[bool]
    radiant_xp_adv: Optional[List[int]]
    # teamfights: Optional[List[TeamfightData]] # pizdec
    teamfights: Optional[TeamfightsData] # TODO: remake with List[TeamfightData]
    tower_status_dire: Optional[int] # хз че за циферка
    tower_status_radiant: Optional[int] # хз че за циферка
    version: Optional[int] # мб версия игры?
    series_id: Optional[int] # no train
    radiant_team: Optional[TeamData]
    dire_team: Optional[TeamData]
    players: Optional[List[InGamePlayerData]] # pizdec
    patch: Optional[int] # кажется версия игры
    throw: Optional[int] # Если DIRE камбекнули, показывает величину камбека (НЕТУ ЧАСТО)
    comeback: Optional[int] # Если RADIANT камбекнули, показывает величину камбека (НЕТУ ЧАСТО)
    # loss: 0 ХЗ НЕТУ ЧАСТО
    # win: 0 ХЗ НЕТУ ЧАСТО
    # replay_url: "string" - (Можно ли вытащить инфу о матче поминутно)

    def __post_init__(self) -> None:
        if self.draft_timings is not None:
            self.draft_timings = [
                DraftItem(
                    order=draft_timing.get('order', None),
                    pick=draft_timing.get('pick', None),
                    active_team=draft_timing.get('active_team', None),
                    hero_id=draft_timing.get('hero_id', None),
                    player_slot=draft_timing.get('player_slot', None),
                    extra_time=draft_timing.get('extra_time', None),
                    total_time_taken=draft_timing.get('total_time_taken', None),
                )
            for draft_timing in self.draft_timings]

        if self.picks_bans is not None:
            self.picks_bans = [
                PickBanItem(
                    is_pick=pick_ban.get('is_pick', None),
                    hero_id=pick_ban.get('hero_id', None),
                    team=pick_ban.get('team', None),
                    order=pick_ban.get('order', None),
                    ord=pick_ban.get('ord', None)
                )
            for pick_ban in self.picks_bans]

        if self.objectives is not None:
            self.objectives = [
                ObjectiveData(
                    time=objective.get('time', None),
                    type=objective.get('type', None),
                    unit=objective.get('unit', None),
                    key=objective.get('key', None),
                    slot=objective.get('slot', None),
                    player_slot=objective.get('player_slot', None),
                    value=objective.get('value', None),
                    killer=objective.get('killer', None),
                    team=objective.get('team', None),
                )
            for objective in self.objectives]
        
        # if self.teamfights is not None:
        #     self.teamfights = None

        # if self.players is not None:
        #     self.players = None