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

    def __init__(self, run_id: str, task_manager=None):
        self.run_id = run_id
        self.task_manager = task_manager
        self.current_config = None
        self.is_collecting = False
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    async def start_passive_listening(self) -> None:
        """Enable passive DTMF listening."""
        self.is_collecting = True
        self.current_config = {
            'termination_key': '#'
        }
        logger.info("DTMF listening enabled with termination_key '#'")

    async def inject_digits_to_conversation(self, digits: str) -> None:
        """Inject digits as user message to conversation history."""
        if not self.task_manager:
            logger.warning("No task_manager for DTMF injection")
            return

        user_message = {"role": "user", "content": digits}
        self.task_manager.history.append(user_message)
        self.task_manager.interim_history.append(user_message)
        logger.info(f"DTMF injected {len(digits)} digits")

def get_dtmf_manager(run_id: str, task_manager=None):
    """Get or create DTMFManager."""
    if run_id not in _dtmf_managers:
        _dtmf_managers[run_id] = DTMFManager(run_id, task_manager)
    return _dtmf_managers[run_id]


def cleanup_dtmf_manager(run_id: str):
    """Remove DTMFManager from cache."""
    if run_id in _dtmf_managers:
        del _dtmf_managers[run_id]
