import time
from typing import Set, Tuple
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class InterruptionManager:
    """Manages interruption state and turn-taking for voice conversations."""

    def __init__(
        self,
        number_of_words_for_interruption: int = 3,
        accidental_interruption_phrases: list = None,
        incremental_delay: int = 900,
        minimum_wait_duration: int = 0,
    ):
        # User speaking state
        self.callee_speaking: bool = False
        self.callee_speaking_start_time: float = -1

        # Sequence management (-1 reserved for background audio)
        self.curr_sequence_id: int = 0
        self.sequence_ids: Set[int] = {-1}

        # Turn tracking
        self.turn_id: int = 0

        # Timing state
        self.let_remaining_audio_pass_through: bool = False
        self.time_since_first_interim_result: float = -1
        self.required_delay_before_speaking: float = 0
        self.incremental_delay: int = incremental_delay
        self.utterance_end_time: float = -1

        # Configuration
        self.number_of_words_for_interruption: int = number_of_words_for_interruption
        self.accidental_interruption_phrases: Set[str] = set(
            accidental_interruption_phrases or []
        )
        self.minimum_wait_duration: int = minimum_wait_duration

        logger.info(
            f"InterruptionManager initialized: "
            f"words_for_interruption={number_of_words_for_interruption}, "
            f"incremental_delay={incremental_delay}ms"
        )

    def get_audio_send_status(self, sequence_id: int, history_length: int = 0) -> str:
        """
        Centralized decision for whether audio should be sent.

        Returns:
        - "SEND" - audio should be sent now
        - "BLOCK" - audio should be discarded (user speaking, invalid sequence)
        - "WAIT" - audio should be delayed (grace period active)
        """
        # Check 1: Invalid sequence - discard
        if sequence_id not in self.sequence_ids:
            return "BLOCK"

        # Check 2: User is speaking - discard (interruption)
        if self.callee_speaking:
            logger.info(f"Audio status=BLOCK - user is speaking")
            return "BLOCK"

        # Check 3: Grace period (only after first 2 turns to avoid latency on welcome)
        if history_length > 2:
            time_since_utterance_end = self.get_time_since_utterance_end()
            if time_since_utterance_end != -1 and time_since_utterance_end < self.incremental_delay:
                logger.info(f"Audio status=WAIT - grace period: {time_since_utterance_end:.0f}ms / {self.incremental_delay}ms")
                return "WAIT"

        return "SEND"

    def should_trigger_interruption(
        self,
        word_count: int,
        transcript: str,
        is_audio_playing: bool,
        welcome_played: bool,
    ) -> bool:
        """Returns True if user speech should trigger an interruption."""
        if not is_audio_playing or not welcome_played:
            return False

        if self.number_of_words_for_interruption == 0:
            return False

        transcript_stripped = transcript.strip()
        return (
            word_count > self.number_of_words_for_interruption
            or transcript_stripped in self.accidental_interruption_phrases
        )

    def is_false_interruption(
        self,
        word_count: int,
        transcript: str,
        is_audio_playing: bool,
        welcome_played: bool,
    ) -> bool:
        """Returns True if final transcript should be ignored (too short, not a phrase match)."""
        if not is_audio_playing or not welcome_played:
            return False

        transcript_stripped = transcript.strip()
        return (
            word_count <= self.number_of_words_for_interruption
            and transcript_stripped not in self.accidental_interruption_phrases
        )

    def on_user_speech_started(self) -> None:
        """Called when user starts speaking (interim transcript received)."""
        if not self.callee_speaking:
            self.callee_speaking = True
            self.callee_speaking_start_time = time.time()
            logger.info(f"User started speaking")

    def on_interim_transcript_received(self) -> None:
        """Called on each interim transcript to update timing state."""
        self.let_remaining_audio_pass_through = False

        if self.time_since_first_interim_result == -1:
            self.time_since_first_interim_result = time.time() * 1000
            logger.info(f"First interim at {self.time_since_first_interim_result}")

    def on_user_speech_ended(self) -> None:
        """Called when user stops speaking (speech_final/UtteranceEnd)."""
        self.callee_speaking = False
        self.let_remaining_audio_pass_through = True
        self.time_since_first_interim_result = -1
        self.utterance_end_time = time.time() * 1000
        logger.info("User speech ended")

    def on_interruption_triggered(self) -> None:
        """Called when interruption cleanup is triggered."""
        self.turn_id += 1
        self.invalidate_pending_responses()
        logger.info(f"Interruption triggered, turn_id={self.turn_id}")

    def invalidate_pending_responses(self) -> None:
        """Invalidates all pending audio by resetting sequence_ids."""
        self.sequence_ids = {-1}
        logger.info("Pending responses invalidated")

    def get_next_sequence_id(self) -> int:
        """Generates and registers a new sequence ID."""
        self.curr_sequence_id += 1
        self.sequence_ids.add(self.curr_sequence_id)
        return self.curr_sequence_id

    def should_delay_output(self, welcome_message_played: bool) -> Tuple[bool, float]:
        """Returns (should_delay, sleep_duration) for incremental delay logic."""
        if not welcome_message_played:
            return False, 0

        return self._check_delay()

    def _check_delay(self) -> Tuple[bool, float]:
        """Check if output should be delayed based on interim result timing."""
        if self.time_since_first_interim_result == -1:
            return False, 0

        elapsed = (time.time() * 1000) - self.time_since_first_interim_result
        if elapsed < self.required_delay_before_speaking:
            return True, 0.1

        return False, 0

    def update_required_delay(self, history_length: int) -> None:
        """Updates delay based on conversation progress (adds delay after first 2 turns)."""
        if history_length > 2:
            self.required_delay_before_speaking = self.incremental_delay
        else:
            self.required_delay_before_speaking = 0

    def reset_delay_for_speech_final(self, history_length: int) -> None:
        """Resets delay variables when speech_final is received."""
        self.time_since_first_interim_result = -1
        if history_length > 2:
            self.required_delay_before_speaking = max(
                self.minimum_wait_duration - self.incremental_delay, 0
            )
        else:
            self.required_delay_before_speaking = 0

    def is_user_speaking(self) -> bool:
        """Returns True if user is currently speaking."""
        return self.callee_speaking

    def is_valid_sequence(self, sequence_id: int) -> bool:
        """Returns True if sequence_id is still valid."""
        return sequence_id in self.sequence_ids

    def get_turn_id(self) -> int:
        """Returns current turn ID."""
        return self.turn_id

    def get_user_speaking_duration(self) -> float:
        """Returns how long user has been speaking in seconds."""
        if not self.callee_speaking or self.callee_speaking_start_time < 0:
            return 0
        return time.time() - self.callee_speaking_start_time

    def get_time_since_utterance_end(self) -> float:
        """Returns time since UtteranceEnd in milliseconds, or -1 if none."""
        if self.utterance_end_time == -1:
            return -1
        return (time.time() * 1000) - self.utterance_end_time

    def reset_utterance_end_time(self) -> None:
        """Resets utterance end time, typically when user continues speaking."""
        self.utterance_end_time = -1
        logger.info("Utterance end time reset")

    def has_pending_responses(self) -> bool:
        """Returns True if there are pending audio responses."""
        return len(self.sequence_ids) > 1

    def set_first_interim_for_immediate_response(self) -> None:
        """Sets first interim timestamp to allow immediate response."""
        self.time_since_first_interim_result = (time.time() * 1000) - 1000
