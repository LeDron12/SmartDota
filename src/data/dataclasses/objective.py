from dataclasses import dataclass

from typing import Dict, List, Union, Optional

@dataclass
class ObjectiveData:
    time: Optional[int] # when objective happened
    type: Optional[str] # Union['building_kill', 'CHAT_MESSAGE_FIRSTBLOOD', 'CHAT_MESSAGE_COURIER_LOST', ...]
    unit: Optional[Union[str, int]] # Union['npc_dota_hero_tiny', 'npc_dota_hero_sniper', -1, ...]
    key: Optional[str] # Union['npc_dota_goodguys_tower2_bot', 'npc_dota_badguys_tower1_bot', ...]
    slot: Optional[int]
    player_slot: Optional[int]
    value: Optional[int]
    killer: Optional[Union[str, int]] # ?
    team: Optional[int] # ?
