import copy

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class MarkEventMetaData:
    def __init__(self):
        self.mark_event_meta_data = {}
        self.previous_mark_event_meta_data = {}
        self.counter = 0

    def update_data(self, mark_id, value):
        value['counter'] = self.counter
        self.counter += 1
        self.mark_event_meta_data[mark_id] = value

    def fetch_data(self, mark_id):
        return self.mark_event_meta_data.pop(mark_id, {})

    def clear_data(self):
        logger.info(f"Clearing mark meta data dict")
        self.counter = 0
        self.previous_mark_event_meta_data = copy.deepcopy(self.mark_event_meta_data)
        self.mark_event_meta_data = {}

    def update_last_post_mark_as_final(self):
        """Update the last stored post-mark to have is_final_chunk=True.
        Returns True if a post-mark was found and updated, False otherwise."""
        last_mark_id = None
        last_counter = -1
        for mark_id, value in self.mark_event_meta_data.items():
            if value.get("type") != "pre_mark_message" and value.get("counter", -1) > last_counter:
                last_counter = value["counter"]
                last_mark_id = mark_id
        if last_mark_id is not None:
            self.mark_event_meta_data[last_mark_id]["is_final_chunk"] = True
            logger.info(f"Updated last post-mark {last_mark_id} (counter={last_counter}) to is_final_chunk=True")
            return True
        return False

    def fetch_cleared_mark_event_data(self):
        return self.previous_mark_event_meta_data

    def __str__(self):
        return f"{self.mark_event_meta_data}"
