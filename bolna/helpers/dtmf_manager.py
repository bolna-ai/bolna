"""
DTMF Manager - Vapi-Style Passive Collection

Always listens for DTMF, injects digits as user messages. No Redis, no polling.
Completes on: termination_key (#) or max_digits (20).
"""
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
_dtmf_managers = {}


class DTMFManager:
    """Passive DTMF: collects digits, injects to conversation on completion."""

    def __init__(self, run_id: str, config, task_manager=None):
        self.run_id = run_id
        self.config = config
        self.task_manager = task_manager
        self.current_config = None
        self.is_collecting = False
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    async def start_passive_listening(self) -> None:
        """Enable passive DTMF listening."""
        self.is_collecting = True
        self.current_config = {
            'max_digits': getattr(self.config, 'max_digits', 20),
            'termination_key': getattr(self.config, 'termination_key', '#')
        }
        logger.info(f"DTMF listening enabled: max_digits={self.current_config['max_digits']}")

    async def inject_digits_to_conversation(self, digits: str) -> None:
        """Inject digits as user message to conversation history."""
        if not self.task_manager:
            logger.warning("No task_manager for DTMF injection")
            return

        user_message = {"role": "user", "content": digits}
        self.task_manager.history.append(user_message)
        self.task_manager.interim_history.append(user_message)
        logger.info(f"DTMF injected {len(digits)} digits")

    def reset(self):
        """Reset collection state."""
        self.is_collecting = False


def get_dtmf_manager(run_id: str, config=None, task_manager=None):
    """Get or create DTMFManager. Returns None if not found and no config."""
    if run_id not in _dtmf_managers:
        if config is None:
            return None
        _dtmf_managers[run_id] = DTMFManager(run_id, config, task_manager)
    return _dtmf_managers[run_id]


def cleanup_dtmf_manager(run_id: str):
    """Remove DTMFManager from cache."""
    if run_id in _dtmf_managers:
        del _dtmf_managers[run_id]
