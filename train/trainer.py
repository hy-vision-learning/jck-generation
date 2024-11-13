from abc import *


class Trainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass
