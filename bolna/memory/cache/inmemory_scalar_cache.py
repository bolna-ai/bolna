from .base_cache import BaseCache
from bolna.helpers.logger_config import configure_logger
import time

logger = configure_logger(__name__)


class InmemoryScalarCache(BaseCache):
    def __init__(self, ttl=-1):
        self.data_dict = {}  # Permanent means permenant only during the duration of the call
        self.ttl_dict = {}
        self.ttl = ttl

    def get(self, key):
        if key in self.data_dict:
            if self.ttl == -1:
                return self.data_dict[key]
            if time.time() - self.ttl_dict[key] < 0:
                return self.data_dict[key]
            # del self.ttl_dict[key]
            # del self.data_dict[key]

        logger.info(f"Cache miss for key {key}")
        return None

    def set(self, key, value):
        self.data_dict[key] = value
        self.ttl_dict[key] = time.time() + self.ttl

    def flush_cache(self, only_ephemeral=True):
        if only_ephemeral:
            self.data_dict.clear()
        self.ttl_dict.clear()
