import pickle
from threading import RLock


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
