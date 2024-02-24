import requests

import sys
sys.path.append('/Users/ankamenskiy/SmartDota/')

from typing import Any, List, Tuple

from src.data.api.OpenDota.base_dataloader import BaseDataloader

from src.data.dataclasses.match import MatchData
from src.data.dataclasses.public_match import PublicMatchData, RANKED_GAME_MODE, RANKED_LOBBY_TYPE
from src.data.dataclasses.team import TeamData
from src.data.dataclasses.teamfight import TeamfightsData
from src.data.dataclasses.player import InGamePlayerData

from collections import deque
from tqdm.notebook import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import logging


class PublicMatchesDataloader(BaseDataloader):

    KEY_PATH = '/Users/ankamenskiy/SmartDota/src/data/api/OpenDota/opendota_api.key'
    API_HOST = "https://api.opendota.com/api"
    MAX_MATCH_INDEX = 9999999998

    LOGS_FILE = '/Users/ankamenskiy/SmartDota/src/data/api/OpenDota/logs/public_matches_downloader.log'

    def __init__(self, 
                 patch_timestamp_start: int, # 1703278800 (23 dec 2023) - 7.35b
                 patch_timestamp_end: int, # 1708549200 (23 feb 2023) - 7.35c
                 num_threads: int=4,
                 batch_size: int=90,
                 verbose: bool=False, 
                 debug: bool=False, 
                 use_key: bool=False
                 ) -> None:
        logging.basicConfig(level=logging.DEBUG, filename=self.LOGS_FILE, filemode="w")

        self.patch_timestamp_start = patch_timestamp_start
        self.patch_timestamp_end = patch_timestamp_end
        self.first_id = self.MAX_MATCH_INDEX

        self.batch_size = batch_size

        self.num_threads = num_threads
        self.verbose = verbose
        self.debug = debug

        self.key = self.__read_key(use_key)
        print(self.key)

        super().__init__()


    def __call__(self, 
                 amount: int,
                 min_rank: int=80,
                 max_rank: int=85
                 ) -> List[MatchData]:

        batch_sizes = [self.batch_size]*(amount // self.batch_size) + [amount % self.batch_size]
        
        for i, batch_size in tqdm(enumerate(batch_sizes), desc='Loading matches batched progress:'):
            if self.verbose:
                logging.info(f'Loading batch: {i + 1}/{len(batch_sizes)}\nBatch size: {batch_size}\n' + '-'*100 + '\n')
                # print(f'Loading batch: {i + 1}/{len(batch_sizes)}\nBatch size: {batch_size}\n', '-'*100, '\n')

            public_matches_data = self.__load_public_matches(batch_size, min_rank, max_rank)
            if self.debug:
                logging.debug('Public matches loaded:\n' + str(len(public_matches_data)) + '\n' + '-'*50)
                # print('Public matches loaded:\n', len(public_matches_data), '\n', '-'*50)
            
            # extended_public_matches_data = self.__load_matches(public_matches_data)
            # if self.debug:
            #     logging.debug('Extended Public matches loaded:\n' + str(len(extended_public_matches_data)) + '\n' + '-'*50)
            #     # print('Public matches loaded:\n', len(extended_public_matches_data), '\n', '-'*50)

            self.data.extend(public_matches_data)
            # self.data.extend(extended_public_matches_data)
            self.save(f'/Users/ankamenskiy/SmartDota/cache/download_checkpoints/public_{len(self.data)}-{amount}.ckpt')

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
    

    def __load_public_matches(self, 
                              amount: int,
                              min_rank: int,
                              max_rank: int
                              ) -> List[PublicMatchData]:

        public_matches = []

        with tqdm(total=amount, desc='Loading public matches data from OpenDota') as pbar:
            while len(public_matches) < amount:

                if self.verbose:
                    logging.info('PUBLIC MATCHES loaded: ' + str(len(public_matches)))
                    logging.info('Totoal requests made: ' + str(self.requests_cnt))
                    # print('PUBLIC MATCHES loaded:', len(public_matches))
                    # print('Totoal requests made:', self.requests_cnt)
                
                mathes_batch = requests.get(
                    url = self.API_HOST + '/publicMatches',
                    params = {
                        'api_key': self.key,
                        'less_than_match_id': self.first_id + 1,
                        'min_rank': min_rank,
                        'max_rank': max_rank
                    }
                ).json()
                self.requests_cnt += 1

                if self.debug:
                    logging.debug(mathes_batch)
                    # print(mathes_batch)
                
                self.first_id = min([elem['match_id'] for elem in mathes_batch])
                # self.first_id = mathes_batch[-1]['match_id']
                public_matches.extend([ 
                    elem for elem in 
                    list(filter( 
                        lambda x: 
                            x.get('game_mode', None) == RANKED_GAME_MODE and x.get('lobby_type', None) == RANKED_LOBBY_TYPE
                            and
                            x.get('start_time', 0) > self.patch_timestamp_start and x.get('start_time', 9_999_999_999) < self.patch_timestamp_end, 
                        mathes_batch 
                    ))
                ]) # Если у команды еще нету id на сервисе, выкидываем

                pbar.update(len(mathes_batch))
                # if self.key == 'Empty Key':
                time.sleep(1)

        public_matches = [
            PublicMatchData(
                match_id=pm.get('match_id', None),
                match_seq_num=pm.get('match_seq_num', None),
                radiant_win=pm.get('radiant_win', None),
                start_time=pm.get('start_time', None),
                duration=pm.get('duration', None),
                lobby_type=pm.get('lobby_type', None),
                game_mode=pm.get('game_mode', None),
                avg_rank_tier=pm.get('avg_rank_tier', None),
                num_rank_tier=pm.get('num_rank_tier', None),
                cluster=pm.get('cluster', None),
                radiant_team=pm.get('radiant_team', None),
                dire_team=pm.get('dire_team', None),
            )
            for pm in public_matches
        ]

        return public_matches[:amount]


    def __load_matches(self, 
                       public_matches_data: List[PublicMatchData], 
                       teams_data: List[Tuple[TeamData, TeamData]]
                       ) -> List[MatchData]:

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
            for i, match_data in enumerate(public_matches_data):
                futures.append(executor.submit(load_match_data, match_id=match_data.match_id, pos=i))

            total = len(public_matches_data)
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
                            public_matches_data[pos] = \
                                MatchData(
                                    match_id=result.get('match_id', None),
                                    public_match_data=public_matches_data[pos],
                                    pro_match_data=None,
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
                                    radiant_team=None,
                                    dire_team=None,
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

        return public_matches_data
