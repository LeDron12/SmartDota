import requests
import pickle
import json

import sys
sys.path.append('/Users/ankamenskiy/SmartDota/')

from typing import Any, List, Tuple
from src.data.dataclasses.match import MatchData
from src.data.dataclasses.pro_match import ProMatchData
from src.data.dataclasses.team import TeamData

from collections import deque
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

    def __init__(self, num_threads: int=4) -> None:
        self.first_id = self.MAX_MATCH_INDEX
        self.num_threads = num_threads
        super().__init__()

    def __call__(self, amount: int) -> List[MatchData]:
        pro_matches_data = self._load_pro_matches(amount)
        print('-'*20)
        print(len(pro_matches_data), pro_matches_data)
        print('Pro matches loaded')
        print('-'*20)
        teams_data = self._load_teams_data(pro_matches_data)
        print('-'*20)
        print(teams_data)
        print('Teams data loaded')
        print('-'*20)
        extended_pro_matches_data = self._load_matches(pro_matches_data, teams_data)
        print('Extended matches data loaded')
        self.data.extend(extended_pro_matches_data)
        return self.data

    def reset_dataloader(self) -> None:
        self.first_id = self.MAX_MATCH_INDEX
        self.data = []

    def _load_pro_matches(self, amount: int) -> List[ProMatchData]:
        pro_matches = []

        with tqdm(total=amount, desc='Loading pro matches data from OpenDota') as pbar:
            while len(pro_matches) < amount:
                print('PRO MATCHES loaded:', len(pro_matches))
                mathes_batch = requests.get(
                    url = self.API_HOST + '/proMatches',
                    params = {
                        'less_than_match_id': self.first_id + 1
                    }
                ).json()
                # print(mathes_batch)
                self.firts_id = mathes_batch[-1]['match_id']
                pro_matches.extend(mathes_batch)

                pbar.update(len(mathes_batch))
                time.sleep(10)

        """ Если у команды еще нету id на сервисе, выкидываем """
        pro_matches = [
            ProMatchData(
                match_id=pm.get('match_id', None),
                radiant_team_id=pm.get('radiant_team_id', None),
                dire_team_id=pm.get('dire_team_id', None),
                leagueid=pm.get('leagueid', None),
                series_type=pm.get('series_type', None),
                radiant_score=pm.get('radiant_score', None),
                dire_score=pm.get('dire_score', None),
                radiant_win=pm.get('radiant_win', None),
            )
            for pm in pro_matches
            # list(          
            #     filter(
            #         lambda x: x.get('radiant_team_id', None) is not None and x.get('dire_team_id', None) is not None, 
            #         pro_matches
            #     )
            # )
        ]

        return pro_matches[:amount]

    def _load_teams_data(self, pro_matches_data: List[ProMatchData]) -> List[Tuple[TeamData, TeamData]]:

        def load_team_data(radiant_team_id: int, dire_team_id: int, pos: int) -> dict:
            try:
                resp_radiant = requests.get(url=self.API_HOST + f'/teams/{radiant_team_id}') #.json()
                resp_radiant = resp_radiant.json() if resp_radiant.text else {}
                resp_dire = requests.get(url=self.API_HOST + f'/teams/{dire_team_id}') #.json()
                resp_dire = resp_dire.json() if resp_dire.text else {}
                return (resp_radiant, resp_dire), (radiant_team_id, dire_team_id) , pos
            except:
                return (None, None), (radiant_team_id, dire_team_id), pos
            
        teams_data = [(None, None)] * len(pro_matches_data)
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = deque()
            for i, match_data in enumerate(pro_matches_data):
                futures.append(executor.submit(load_team_data, radiant_team_id=match_data.radiant_team_id, dire_team_id=match_data.dire_team_id, pos=i))

            total = len(pro_matches_data)
            with tqdm(total=total, desc='Loading matches data from OpenDota') as pbar: 
                while len(futures) > 0:
                    print('TEAMS remained futures:', len(futures))
                    for future in tqdm(as_completed(futures), desc='Loading teams data from OpenDota'):
                        futures.popleft()
                        results, team_ids, pos = future.result()

                        if results[0] is None or 'error' in results[0] or results[1] is None or 'error' in results[1]:
                            futures.append(executor.submit(load_team_data, radiant_team_id=team_ids[0], dire_team_id=team_ids[1], pos=pos))
                        else:
                            teams_data[pos] = (
                                TeamData( # Radiant
                                    tag=results[0].get('tag', None),
                                    name=results[0].get('name', None),
                                    team_id=results[0].get('team_id', None),
                                    rating=results[0].get('rating', None),
                                    wins=results[0].get('wins', None),
                                    losses=results[0].get('losses', None),
                                    last_match_time=results[0].get('last_match_time', None),
                                ),
                                TeamData( # Dire
                                    tag=results[1].get('tag', None),
                                    name=results[1].get('name', None),
                                    team_id=results[1].get('team_id', None),
                                    rating=results[1].get('rating', None),
                                    wins=results[1].get('wins', None),
                                    losses=results[1].get('losses', None),
                                    last_match_time=results[1].get('last_match_time', None),
                                )
                            )

                    pbar.update(total - len(futures))
                    if len(futures) > 0:
                        time.sleep(45)

        return teams_data

    def _load_matches(self, pro_matches_data: List[ProMatchData], teams_data: List[Tuple[TeamData, TeamData]]) -> List[MatchData]:

        def load_match_data(match_id: int, pos: int) -> dict:
            try:
                resp = requests.get(url=self.API_HOST + f'/matches/{match_id}').json()
                return resp, match_id, pos
            except:
                return None, match_id, pos
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = deque()
            for i, match_data in enumerate(pro_matches_data):
                futures.append(executor.submit(load_match_data, match_id=match_data.match_id, pos=i))

            total = len(pro_matches_data)
            with tqdm(total=total, desc='Loading matches data from OpenDota') as pbar: 
                while len(futures) > 0:
                    print('MATCHES remained futures:', len(futures))
                    for future in tqdm(as_completed(futures), desc='Loading extra data from OpenDota'):
                        futures.popleft()
                        result, match_id, pos = future.result()

                        if result is None or 'error' in result:
                            futures.append(executor.submit(load_match_data, match_id=match_id, pos=pos))
                        else:
                            pro_matches_data[pos] = MatchData(
                                                        match_id=result.get('match_id', None),
                                                        pro_match_data=pro_matches_data[pos],
                                                        barracks_status_dire=result.get('barracks_status_dire', None),
                                                        barracks_status_radiant=result.get('barracks_status_radiant', None),
                                                        dire_score=result.get('dire_score', None),
                                                        draft_timings=result.get('draft_timings', None),
                                                        picks_bans=result.get('picks_bans', None),
                                                        duration=result.get('duration', None),
                                                        first_blood_time=result.get('first_blood_time', None),
                                                        leagueid=result.get('leagueid', None),
                                                        objectives=result.get('objectives', None),
                                                        radiant_gold_adv=result.get('radiant_gold_adv', None),
                                                        radiant_score=result.get('radiant_score', None),
                                                        radiant_win=result.get('radiant_win', None),
                                                        radiant_xp_adv=result.get('radiant_xp_adv', None),
                                                        teamfights=result.get('teamfights', None),
                                                        tower_status_dire=result.get('tower_status_dire', None),
                                                        tower_status_radiant=result.get('tower_status_radiant', None),
                                                        version=result.get('version', None),
                                                        series_id=result.get('series_id', None),
                                                        radiant_team=teams_data[pos][0],
                                                        dire_team=teams_data[pos][1],
                                                        players=result.get('players', None),
                                                        patch=result.get('patch', None),
                                                        throw=result.get('throw', None),
                                                        comeback=result.get('comeback', None)
                                                    )

                    pbar.update(total - len(futures))
                    if len(futures) > 0:
                        time.sleep(45)

        return pro_matches_data
        