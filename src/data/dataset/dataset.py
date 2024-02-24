import sys
sys.path.append('/Users/ankamenskiy/SmartDota/')

from typing import List
from src.data.dataclasses.match import MatchData

from torch.utils.data import Dataset, DataLoader

import json

import pandas as pd
import numpy as np
import torch


# class DotaDataset:

#     def __init__(self, config) -> None:
#         super().__init__()

#         self.data = data

#         if isinstance(config, str):
#             config = json.loads(config)
#         self.config = config       


class DraftDataset(Dataset):

    def __init__(self, data, config) -> None:
        super().__init__()

        self.data = self._make_dataset(data)

        if isinstance(config, str):
            config = json.loads(config)
        self.config = config['draft']        

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _make_dataset(self, data: List[MatchData]):
        dataset = []
        # heroes_count = max([pick_ban.hero_id for pick_ban in data.picks_bans])
        
        for elem in data:
            row = {}

            if self.config['hero_stats']['id']:
                if 
                row['radiant_hero_ids'] = dataset.picks_bans
            # dataset.append([])
            # elem =  elem.__dict__

            # # features
            # if self.config['hero_stats']['id']:
            #     picks = [
            #         pick_ban['hero_id'] + 1 if pick_ban['team'] == 1 else -pick_ban['hero_id']
            #         for pick_ban 
            #         in list(filter(lambda x: x['is_pick'], elem['picks_bans']))
            #     ]
            #     one_hot_picks = [] * heroes_count
            #     for pick in picks:
            #         sign = 1 if pick > 0 else -1
            #         one_hot_picks[pick * sign - 1] = sign
            #     dataset[-1].extend(one_hot_picks)

            # # target
            # dataset[-1].append(elem.pro_match_data.radiant_win)
        
        return dataset

    def as_pandas(self):
        data = np.array(self.data, dtype=np.int32)
        features, target = data[:, :-1], data[:, -1]

        zeros = np.zeros((features.size, features.max() + 1), dtype=np.int32)
        data = zeros[np.arange(features.size, dtype=np.int32), features] = 1

        df = pd.DataFrame(self.data)
        return df
