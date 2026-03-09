from collections import OrderedDict
from .base_cache import BaseCache
from bolna.helpers.logger_config import configure_logger
import time
logger = configure_logger(__name__)

class InmemoryScalarCache(BaseCache):
    def __init__(self, ttl=-1, max_size=200):
        self.data_dict = OrderedDict()
        self.ttl_dict = {}
        self.ttl = ttl
        self.max_size = max_size

    def get(self, key):
        if key in self.data_dict:
            if self.ttl == -1:
                self.data_dict.move_to_end(key)
                return self.data_dict[key]
            if time.time() - self.ttl_dict[key] < self.ttl:
                self.data_dict.move_to_end(key)
                return self.data_dict[key]
            # TTL expired — evict
            del self.ttl_dict[key]
            del self.data_dict[key]

        logger.info(f"Cache miss for key {key}")
        return None

    def set(self, key, value):
        if key in self.data_dict:
            self.data_dict.move_to_end(key)
        self.data_dict[key] = value
        self.ttl_dict[key] = time.time()
        # Evict oldest if over max_size
        while len(self.data_dict) > self.max_size:
            oldest_key, _ = self.data_dict.popitem(last=False)
            self.ttl_dict.pop(oldest_key, None)

    def flush_cache(self, only_ephemeral=True):
        if only_ephemeral:
            self.data_dict.clear()
        self.ttl_dict.clear()
