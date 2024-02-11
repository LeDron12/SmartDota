import sys
sys.path.append('/home/ankamenskiy/SmartDota/')

from src.data.api.download import ProMatchesDataloader

TOTAL_AMOUNT = 1_000
# TOTAL_AMOUNT = 10
path = f'../../cache/pro_{TOTAL_AMOUNT}'

downloader = ProMatchesDataloader(4)
matches = downloader(TOTAL_AMOUNT)
downloader.save(path)
downloader.load(path)

print(downloader.data)