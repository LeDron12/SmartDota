import requests
from tqdm import tqdm

import sys
sys.path.append('/Users/ankamenskiy/SmartDota')

from src.data.dataclasses.match import MatchData
from typing import Any, List


class ProMatchesDataDownloader:

    API_HOST = "https://api.opendota.com/api"
    MAX_MATCH_INDEX = 9999999998

    def __init__(self) -> None:
        self.first_id = self.MAX_MATCH_INDEX

    # TODO: add multithreading load
    def __call__(self, amount: int, *args: Any, **kwds: Any) -> List[MatchData]:
        pro_matches_data = self._load_data(amount)
        self._extend_data(pro_matches_data)
        return self._convert_data(pro_matches_data)

    def reset_load_index(self) -> None:
        self.first_id = self.MAX_MATCH_INDEX

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
        for match_data in tqdm(matches_data, desc='Loading extra data from OpenDota'):
            extended_data = requests.get(
                url = self.API_HOST + f'/matches/{match_data["match_id"]}'
            ).json()
            match_data.update(extended_data)

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
