from abc import ABC, abstractmethod


class BaseCache(ABC):
    @abstractmethod
    def set(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        pass
