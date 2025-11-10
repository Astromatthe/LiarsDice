from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, pid: int):
        self.pid = pid

    @abstractmethod
    def act(self, game):
        pass