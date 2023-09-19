import sys
sys.path.append('/Users/ankamenskiy/SmartDota')

from src.data.api.download import ProMatchesDataloader

downloader = ProMatchesDataloader()
matches = downloader(5)
downloader.save('../../cache/pro_5')
downloader.load('../../cache/pro_5')

print(downloader.data)