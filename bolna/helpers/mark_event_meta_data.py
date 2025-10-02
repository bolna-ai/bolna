import copy

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class MarkEventMetaData:
    def __init__(self):
        self.mark_event_meta_data = {}
        self.previous_mark_event_meta_data = {}
        self.counter = 0

    def update_data(self, mark_id, value):
        if value.get('type') != 'ambient_noise':
            logger.info(f"Updating mark_id = {mark_id} with value = {value}")
        value['counter'] = self.counter
        self.counter += 1
        self.mark_event_meta_data[mark_id] = value

    def fetch_data(self, mark_id):
        data = self.mark_event_meta_data.pop(mark_id, {})
        if data.get('type') != 'ambient_noise':
            logger.info(f"Fetching meta data details for mark_id = {mark_id}")
        return data

    def clear_data(self):
        logger.info(f"Clearing mark meta data dict")
        self.counter = 0
        self.previous_mark_event_meta_data = copy.deepcopy(self.mark_event_meta_data)
        self.mark_event_meta_data = {}

    def fetch_cleared_mark_event_data(self):
        return self.previous_mark_event_meta_data

    def __str__(self):
        return f"{self.mark_event_meta_data}"
