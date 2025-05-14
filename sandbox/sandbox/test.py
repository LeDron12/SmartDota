import sys
sys.path.append('/Users/ankamenskiy/SmartDota/')

from src.data.api.download import ProMatchesDataloader


downloader = ProMatchesDataloader(num_threads=4, verbose=True, use_key=True)

TOTAL_AMOUNT_LOAD = 1_000
path = f'../../cache/pro_{TOTAL_AMOUNT_LOAD}'
downloader.load(path)

TOTAL_AMOUNT_SAVE = 5_000 # 1000 + 4000
path = f'../../cache/pro_{TOTAL_AMOUNT_SAVE}'
matches = downloader(TOTAL_AMOUNT_SAVE - TOTAL_AMOUNT_LOAD)
downloader.save(path)

print('Loaded matches: ', len(downloader.data))
print('Requests count: ', downloader.requests_cnt)