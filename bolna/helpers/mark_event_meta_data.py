from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

class MarkEventMetaData:
    def __init__(self):
        self.mark_event_meta_data = {}

    def update_data(self, mark_id, value):
        logger.info(f"Updating mark_id = {mark_id} with value = {value}")
        self.mark_event_meta_data[mark_id] = value

    def fetch_data(self, mark_id):
        logger.info(f"Fetching meta data details for mark_id = {mark_id}")
        return self.mark_event_meta_data.pop(mark_id, {})

    def clear_data(self):
        logger.info(f"Clearing mark meta data dict")
        self.mark_event_meta_data = {}

    def __str__(self):
        return f"{self.mark_event_meta_data}"