import requests
import pickle

import sys
sys.path.append('/Users/ankamenskiy/SmartDota')

from src.data.dataclasses.match import MatchData
from typing import Any, List
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
import time


class BaseDataloader:

    def __init__(self) -> None:
        self.lock = RLock()
        self.data = []

    def reset_dataloader(self) -> None:
        self.data = [] 

    def save(self, path: str):
        pickle.dump(self.data, open(path, "wb"))

    def load(self, path: str):
        self.data = pickle.load(open(path, "rb"))


class ProMatchesDataloader(BaseDataloader):

    API_HOST = "https://api.opendota.com/api"
    MAX_MATCH_INDEX = 9999999998

    def __init__(self, num_threads: int = 4) -> None:
        self.first_id = self.MAX_MATCH_INDEX
        self.num_threads = num_threads
        super().__init__()

    def __call__(self, amount: int, *args: Any, **kwds: Any) -> List[MatchData]:
        pro_matches_data = self._load_data(amount)
        self._extend_data(pro_matches_data)
        self.data.extend(self._convert_data(pro_matches_data))
        return self.data

    def reset_dataloader(self) -> None:
        self.first_id = self.MAX_MATCH_INDEX
        self.data = []

    def _load_data(self, amount: int) -> list:
        pro_matches = []

        with tqdm(total=amount, desc='Loading pro matches data from OpenDota') as pbar:
            while len(pro_matches) < amount:
                mathes_batch = requests.get(
                    url = self.API_HOST + '/proMatches',
                    params = {
                        'less_than_match_id': self.first_id + 1
                    }
                ).json()
                self.firts_id = mathes_batch[-1]['match_id']
                pro_matches.extend(mathes_batch)

                pbar.update(len(mathes_batch))

        return pro_matches[:amount]


    def _extend_data(self, matches_data: list) -> None:

        def load_match_data(match_id: int) -> dict:
            return requests.get(url=self.API_HOST + f'/matches/{match_id}').json()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for match_data in matches_data:
                futures.append(executor.submit(load_match_data, match_id=match_data['match_id']))
            for i, future in tqdm(enumerate(as_completed(futures)), desc='Loading extra data from OpenDota'):
                matches_data[i].update(future.result())

    def _convert_data(self, matches_data: list) -> List[MatchData]:

        for match_data in tqdm(matches_data, desc='Filtering and converting data'):
            picks_only = filter(lambda x: x['is_pick'], match_data['picks_bans'])
            match_data['picks'] = [{
                'order': pick['order'],
                'hero_id': pick['hero_id'], 
                'team_id': match_data['dire_team_id'] if pick['team'] == 1 else match_data['radiant_team_id']
            } for pick in picks_only]

        converted_data = [
            MatchData(
                match_id=match['match_id'],
                radiant_team_id=match['radiant_team_id'],
                dire_team_id=match['dire_team_id'],
                picks=match['picks'],
                patch_id=match['patch']
            )
        for match in matches_data]

        return converted_data
