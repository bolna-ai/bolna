from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseAgent:
    def __init__(self):
        self.agent_name = "base-agent"

    def silence_exit_pending(self) -> bool:
        """True while the current node needs more [silence] turns to reach its own silence exit edge."""
        return False
