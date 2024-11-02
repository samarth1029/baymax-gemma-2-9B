from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def prepare_for_training(self):
        pass
