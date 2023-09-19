import sys
sys.path.append('/Users/ankamenskiy/SmartDota')

from src.data.api.download import ProMatchesDataDownloader

downloader = ProMatchesDataDownloader()
matches = downloader(40)

print(matches)