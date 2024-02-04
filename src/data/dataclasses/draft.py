from dataclasses import dataclass

from typing import Dict, List, Union, Optional

@dataclass
class DraftItem:
    order: int
    pick: bool
    active_team: int
    hero_id: int
    player_slot: int
    extra_time: int
    total_time_taken: int

@dataclass
class PickBanItem:
    is_pick: bool
    hero_id: int
    team: int # 0 or 1
    order: int # == ord
    ord: int # == order
