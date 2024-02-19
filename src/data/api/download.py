import requests
import pickle
import json

import os
import sys
sys.path.append('/Users/ankamenskiy/SmartDota/')

from typing import Any, List, Tuple
from src.data.dataclasses.match import MatchData
from src.data.dataclasses.pro_match import ProMatchData
from src.data.dataclasses.team import TeamData
from src.data.dataclasses.teamfight import TeamfightsData
from src.data.dataclasses.player import InGamePlayerData

from collections import deque
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
import time


class BaseDataloader:

    def __init__(self) -> None:
        self.lock = RLock()
        self.data = []

        self.requests_cnt = 0

    def reset_dataloader(self) -> None:
        self.data = []

    def save(self, path: str):
        pickle.dump(self.data, open(path, "wb"))

    def load(self, path: str):
        self.data = pickle.load(open(path, "rb"))


class ProMatchesDataloader(BaseDataloader):

    KEY_PATH = '/Users/ankamenskiy/SmartDota/src/data/api/api.key'
    API_HOST = "https://api.opendota.com/api"
    MAX_MATCH_INDEX = 9999999998

    def __init__(self, 
                 num_threads: int=4,
                 batch_size: int=90,
                 verbose: bool=False, 
                 debug: bool=False, 
                 use_key: bool=False
                 ) -> None:
        self.first_id = self.MAX_MATCH_INDEX
        self.batch_size = batch_size

        self.num_threads = num_threads
        self.verbose = verbose
        self.debug = debug

        self.key = self.__read_key(use_key)
        print(self.key)

        super().__init__()


    def __call__(self, amount: int) -> List[MatchData]:

        batch_sizes = [self.batch_size]*(amount // self.batch_size) + [amount % self.batch_size]
        
        for i, batch_size in tqdm(enumerate(batch_sizes), desc='Loading matches batched progress:'):
            if self.verbose:
                print(f'Loading batch: {i + 1}/{len(batch_sizes)}\nBatch size: {batch_size}\n', '-'*100, '\n')

            pro_matches_data = self.__load_pro_matches(amount)
            if self.verbose:
                print('-'*20, '\n', len(pro_matches_data), '\n', pro_matches_data, '\n', 'Pro matches loaded', '\n', '-'*20)

            teams_data = self.__load_teams_data(pro_matches_data)
            if self.verbose:
                print('-'*20, '\n', len(teams_data), '\n', teams_data, '\n', 'Teams data loaded', '\n', '-'*20)
        
            extended_pro_matches_data = self.__load_matches(pro_matches_data, teams_data)
            print('Extended matches data loaded')
        
            self.data.extend(extended_pro_matches_data)
            self.save(f'/Users/ankamenskiy/SmartDota/cache/download_checkpoints/pro_{len(self.data)}-{amount}.ckpt')

        return self.data
    

    def reset_dataloader(self) -> None:
        self.first_id = self.MAX_MATCH_INDEX
        self.data = []
        self.requests_cnt = 0


    def load(self, path: str):
        super().load(path)
        self.MAX_MATCH_INDEX = min([elem.match_id for elem in self.data])
        print('Last match index:', self.MAX_MATCH_INDEX)
    

    def __read_key(self, use_key):
        if use_key:
            with open(self.KEY_PATH, 'r') as f:
                return f.read()
        return None


    def __load_pro_matches(self, amount: int) -> List[ProMatchData]:

        pro_matches = []

        with tqdm(total=amount, desc='Loading pro matches data from OpenDota') as pbar:
            while len(pro_matches) < amount:

                if self.verbose:
                    print('PRO MATCHES loaded:', len(pro_matches))
                    print('Totoal requests made:', self.requests_cnt)
                
                mathes_batch = requests.get(
                    url = self.API_HOST + '/proMatches',
                    params = {
                        'api_key': self.key,
                        'less_than_match_id': self.first_id + 1
                    }
                ).json()
                self.requests_cnt += 1

                if self.debug:
                    print(mathes_batch)
                
                self.first_id = mathes_batch[-1]['match_id']
                pro_matches.extend([ 
                    elem for elem in 
                    list(filter( 
                        lambda x: x.get('radiant_team_id', None) is not None and x.get('dire_team_id', None) is not None, 
                        mathes_batch 
                    ))
                ]) # Если у команды еще нету id на сервисе, выкидываем

                pbar.update(len(mathes_batch))
                time.sleep(10)

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
        ]

        return pro_matches[:amount]


    def __load_teams_data(self, pro_matches_data: List[ProMatchData]) -> List[Tuple[TeamData, TeamData]]:
        
        def request(team_id):
            resp = requests.get(
                        url=self.API_HOST + f'/teams/{team_id}',
                        params={'api_key': self.key}
                    )
            self.requests_cnt += 1
            return resp

        def load_team_data(radiant_team_id: int, dire_team_id: int, pos: int) -> dict:
            try:
                resp_radiant = request(radiant_team_id) #.json()
                resp_radiant = resp_radiant.json() if resp_radiant.text else {}
                if self.debug:
                    print('radiant_team_id:', radiant_team_id, '\n' ,'resp_radiant:', resp_radiant, '\n', '-'*20)

                resp_dire = request(dire_team_id) #.json()
                resp_dire = resp_dire.json() if resp_dire.text else {}
                if self.debug:
                    print('dire_team_id:', dire_team_id, '\n' ,'resp_dire:', resp_dire, '\n', '-'*20)
                
                return (resp_radiant, resp_dire), (radiant_team_id, dire_team_id) , pos
            except:
                return (None, None), (radiant_team_id, dire_team_id), pos
            

        teams_data = [(None, None)] * len(pro_matches_data)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = deque()
            for i, match_data in enumerate(pro_matches_data):
                futures.append(executor.submit(load_team_data, radiant_team_id=match_data.radiant_team_id, dire_team_id=match_data.dire_team_id, pos=i))

            total = len(pro_matches_data)
            with tqdm(total=total, desc='Loading teams data from OpenDota') as pbar: 
                while len(futures) > 0:

                    if self.verbose:
                        print('TEAMS remained futures:', len(futures))
                        print('Totoal requests made:', self.requests_cnt)
                    
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


    def __load_matches(self, pro_matches_data: List[ProMatchData], teams_data: List[Tuple[TeamData, TeamData]]) -> List[MatchData]:

        def load_match_data(match_id: int, pos: int) -> dict:
            try:
                resp = requests.get(
                    url=self.API_HOST + f'/matches/{match_id}',
                    params={'api_key': self.key}
                ).json()
                self.requests_cnt += 1
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

                    if self.verbose:
                        print('MATCHES remained futures:', len(futures))
                        print('Totoal requests made:', self.requests_cnt)
                    
                    for future in tqdm(as_completed(futures), desc='Loading extra data from OpenDota'):
                        futures.popleft()
                        result, match_id, pos = future.result()

                        if result is None or 'error' in result:
                            futures.append(executor.submit(load_match_data, match_id=match_id, pos=pos))
                        else:
                            if self.debug:
                                print("result.get('players', [])", result.get('players', []))
                                print("result.get('teamfights', [])", result.get('teamfights', []))
                            pro_matches_data[pos] = \
                                MatchData(
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
                                    # teamfights=result.get('teamfights', None),
                                    teamfights=TeamfightsData(
                                        result.get('teamfights', [])
                                    ),
                                    tower_status_dire=result.get('tower_status_dire', None),
                                    tower_status_radiant=result.get('tower_status_radiant', None),
                                    version=result.get('version', None),
                                    series_id=result.get('series_id', None),
                                    radiant_team=teams_data[pos][0],
                                    dire_team=teams_data[pos][1],
                                    # players=result.get('players', None),
                                    players=[
                                        InGamePlayerData(player=p)
                                        for p
                                        in result.get('players', [])
                                    ],
                                    patch=result.get('patch', None),
                                    throw=result.get('throw', None),
                                    comeback=result.get('comeback', None)
                                )

                        pbar.update(total - len(futures))
                    if len(futures) > 0:
                        time.sleep(45)

        return pro_matches_data
        