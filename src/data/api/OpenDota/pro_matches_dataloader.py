import requests

import sys
sys.path.append('/home/david/SmartDota/')

from typing import Any, List, Tuple

from src.data.api.OpenDota.base_dataloader import BaseDataloader

from src.data.dataclasses.match import MatchData
from src.data.dataclasses.pro_match import ProMatchData
from src.data.dataclasses.team import TeamData
from src.data.dataclasses.teamfight import TeamfightsData
from src.data.dataclasses.player import InGamePlayerData

from collections import deque
from tqdm.notebook import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import logging


class ProMatchesDataloader(BaseDataloader):

    # DEBUG_LOGS_FILE = '/home/david/SmartDota/src/data/api/OpenDota/pro_matches_downloader_log.debug'
    # INFO_LOGS_FILE = '/home/david/SmartDota/src/data/api/OpenDota/pro_matches_downloader_log.info'
    LOGS_FILE = '/home/david/SmartDota/src/data/api/OpenDota/logs/pro_matches_downloader.log'

    KEY_PATH = '/home/david/SmartDota/src/data/api/OpenDota/opendota_api.key'
    API_HOST = "https://api.opendota.com/api"
    MAX_MATCH_INDEX = 9999999998

    FAILS_THRESHOLD = 3

    def __init__(self, 
                 num_threads: int=4,
                 batch_size: int=90,
                 verbose: bool=False, 
                 debug: bool=False, 
                 use_key: bool=False
                 ) -> None:
        # logging.basicConfig(level=logging.DEBUG, filename=self.DEBUG_LOGS_FILE, filemode="w")
        # logging.basicConfig(level=logging.INFO, filename=self.INFO_LOGS_FILE, filemode="w")
        logging.basicConfig(level=logging.DEBUG, filename=self.LOGS_FILE, filemode="w")

        self.first_id = self.MAX_MATCH_INDEX
        self.batch_size = batch_size

        self.num_threads = num_threads
        self.verbose = verbose
        self.debug = debug

        self.key = self.__read_key(use_key).strip()
        logging.info(str(self.key))
        # print(self.key)

        self.teams_cache = {}

        super().__init__()


    def __call__(self, amount: int) -> List[MatchData]:

        batch_sizes = [self.batch_size]*(amount // self.batch_size) + [amount % self.batch_size]
        
        for i, batch_size in tqdm(enumerate(batch_sizes), desc='Loading matches batched progress:'):
            if self.verbose:
                logging.info(f'Loading batch: {i + 1}/{len(batch_sizes)}\nBatch size: {batch_size}\n' + '-'*100 + '\n')
                # print(f'Loading batch: {i + 1}/{len(batch_sizes)}\nBatch size: {batch_size}\n', '-'*100, '\n')

            pro_matches_data = self.__load_pro_matches(batch_size)
            if self.debug:
                logging.debug('-'*20 + '\n' + str(len(pro_matches_data)) + '\n' + str(pro_matches_data) + '\n' + 'Pro matches loaded' + '\n' + '-'*20)
                # print('-'*20, '\n', len(pro_matches_data), '\n', pro_matches_data, '\n', 'Pro matches loaded', '\n', '-'*20)

            teams_data = self.__load_teams_data(pro_matches_data)
            if self.debug:
                logging.debug('-'*20 + '\n' + str(len(teams_data)) + '\n' + str(teams_data) + '\n' + 'Teams data loaded' + '\n' + '-'*20)
                # print('-'*20, '\n', len(teams_data), '\n', teams_data, '\n', 'Teams data loaded', '\n', '-'*20)
        
            extended_pro_matches_data = self.__load_matches(pro_matches_data, teams_data)
            logging.info('Extended matches data loaded')
            # print('Extended matches data loaded')
        
            self.data.extend(extended_pro_matches_data)
            self.save(f'/home/david/SmartDota/cache/download_checkpoints/pro_{len(self.data)}-{amount}.ckpt')

        return self.data
    

    def reset_dataloader(self) -> None:
        self.first_id = self.MAX_MATCH_INDEX
        self.data = []
        self.requests_cnt = 0


    def load(self, path: str):
        super().load(path)
        self.first_id = min([elem.match_id for elem in self.data])

        logging.info('Last match index: ' +  str(self.first_id))
        print('Last match index:', self.first_id)
    

    def __read_key(self, use_key):
        if use_key:
            with open(self.KEY_PATH, 'r') as f:
                return f.read()
        return 'Empty Key'


    def __load_pro_matches(self, amount: int) -> List[ProMatchData]:

        pro_matches = []

        with tqdm(total=amount, desc='Loading pro matches data from OpenDota') as pbar:
            while len(pro_matches) < amount:

                if self.verbose:
                    logging.info('PRO MATCHES loaded: ' + str(len(pro_matches)))
                    logging.info('Totoal requests made: ' + str(self.requests_cnt))
                    # print('PRO MATCHES loaded:', len(pro_matches))
                    # print('Totoal requests made:', self.requests_cnt)
                
                mathes_batch = requests.get(
                    url = self.API_HOST + '/proMatches',
                    params = {
                        'api_key': self.key,
                        'less_than_match_id': self.first_id + 1
                    }
                ).json()
                self.requests_cnt += 1

                if self.debug:
                    logging.debug(mathes_batch)
                    # print(mathes_batch)
                
                # self.first_id = mathes_batch[-1]['match_id']
                self.first_id = min([elem['match_id'] for elem in mathes_batch])
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
                duration=pm.get('duration', None),
                start_time=pm.get('start_time', None),
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

        fails_count = {}
        
        def request(team_id):
            resp = requests.get(
                        url=self.API_HOST + f'/teams/{team_id}',
                        params={'api_key': self.key}
                    )
            self.requests_cnt += 1
            return resp

        def load_team_data(radiant_team_id: int, dire_team_id: int, pos: int) -> dict:
            try:
                if radiant_team_id in self.teams_cache and self.teams_cache[radiant_team_id] != {}:
                    resp_radiant = self.teams_cache[radiant_team_id]
                else:
                    resp_radiant = request(radiant_team_id) #.json()
                    resp_radiant = resp_radiant.json() if resp_radiant.text else {}
                    self.teams_cache[radiant_team_id] = resp_radiant
                if self.debug:
                    logging.debug('radiant_team_id: ' + str(radiant_team_id) + '\n' + 'resp_radiant:' + str(resp_radiant) + '\n' + '-'*20)
                    # print('radiant_team_id:', radiant_team_id, '\n' ,'resp_radiant:', resp_radiant, '\n', '-'*20)
                
                if dire_team_id in self.teams_cache and self.teams_cache[dire_team_id] != {}:
                    resp_dire = self.teams_cache[dire_team_id]
                else:
                    resp_dire = request(dire_team_id) #.json()
                    resp_dire = resp_dire.json() if resp_dire.text else {}
                    self.teams_cache[dire_team_id] = resp_dire
                if self.debug:
                    logging.debug('dire_team_id: ' + str(dire_team_id) + '\n' + 'resp_dire:' + str(resp_dire) + '\n' + '-'*20)
                    # print('dire_team_id:', dire_team_id, '\n' ,'resp_dire:', resp_dire, '\n', '-'*20)
                
                return (resp_radiant, resp_dire), (radiant_team_id, dire_team_id) , pos
            except:
                if self.verbose:
                    logging.info('Failed loading:\n' + f'radiant_team_id: {radiant_team_id}\ndire_team_id: {dire_team_id}')
                    # print('Failed loading:\n', f'radiant_team_id: {radiant_team_id}\ndire_team_id: {dire_team_id}')
                fails_count[radiant_team_id] = fails_count.get(radiant_team_id, 0) + 1
                fails_count[dire_team_id] = fails_count.get(dire_team_id, 0) + 1
                
                if fails_count[radiant_team_id] > self.FAILS_THRESHOLD or fails_count[dire_team_id] > self.FAILS_THRESHOLD:
                    fails_count[radiant_team_id] = 0
                    fails_count[dire_team_id] = 0
                    return ({}, {}), (radiant_team_id, dire_team_id), pos
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
                        logging.info('TEAMS remained futures:' + str(len(futures)) + '\n' + 'Totoal requests made:' + str(self.requests_cnt) + '\n' + '-'*50)
                        # print('TEAMS remained futures:', len(futures))
                        # print('Totoal requests made:', self.requests_cnt)
                        # print('-'*50)
                    
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

        fails_count = {}

        def load_match_data(match_id: int, pos: int) -> dict:
            try:
                resp = requests.get(
                    url=self.API_HOST + f'/matches/{match_id}',
                    params={'api_key': self.key}
                ).json()
                self.requests_cnt += 1
                return resp, match_id, pos
            except:
                if self.verbose:
                    logging.info('Failed loading:\n' + f'match_id: {match_id}')
                    # print('Failed loading:\n', f'match_id: {match_id}')
                fails_count[match_id] = fails_count.get(match_id, 0) + 1
                
                if fails_count[match_id] > self.FAILS_THRESHOLD:
                    fails_count[match_id] = 0
                    return {}, match_id, pos
                return None, match_id, pos
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = deque()
            for i, match_data in enumerate(pro_matches_data):
                futures.append(executor.submit(load_match_data, match_id=match_data.match_id, pos=i))

            total = len(pro_matches_data)
            with tqdm(total=total, desc='Loading matches data from OpenDota 1') as pbar: 
                while len(futures) > 0:

                    if self.verbose:
                        logging.info('MATCHES remained futures: ' + str(len(futures)) + '\n' + 'Totoal requests made: ' + str(self.requests_cnt))
                        # print('MATCHES remained futures:', len(futures))
                        # print('Totoal requests made:', self.requests_cnt)
                    
                    for future in tqdm(as_completed(futures), desc='Loading matches data from OpenDota 2'):
                        futures.popleft()
                        result, match_id, pos = future.result()

                        if result is None or 'error' in result:
                            futures.append(executor.submit(load_match_data, match_id=match_id, pos=pos))
                        else:
                            if self.debug:
                                logging.debug("result.get('players', []):\n" + str(result.get('players', [])) + '\n' + '-'*30 + '\n' + "result.get('teamfights', []):\n" + str(result.get('teamfights', [])))
                                # print("result.get('players', [])", result.get('players', []))
                                # print("result.get('teamfights', [])", result.get('teamfights', []))
                            pro_matches_data[pos] = \
                                MatchData(
                                    match_id=result.get('match_id', None),
                                    public_match_data=None,
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
