import time
from typing import Set, Tuple
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class InterruptionManager:
    """
    Manages interruption state and decision-making for voice conversations.

    Responsibilities:
    - Track user speaking state (callee_speaking)
    - Manage sequence IDs for response invalidation
    - Decide when to trigger interruption cleanup
    - Decide when audio should be sent to user
    - Track turn IDs and timing state

    This class centralizes all interruption-related logic that was previously
    scattered across TaskManager, making it easier to maintain and test.
    """

    def __init__(
        self,
        number_of_words_for_interruption: int = 3,
        accidental_interruption_phrases: list = None,
        incremental_delay: int = 900,
    ):
        """
        Initialize the InterruptionManager.

        Args:
            number_of_words_for_interruption: Minimum word count to trigger real interruption.
                                              Set to 0 to disable interruption detection.
            accidental_interruption_phrases: List of phrases that should always trigger
                                             interruption regardless of word count.
            incremental_delay: Delay in milliseconds before responding after first interim result.
        """
        # User speaking state
        self.callee_speaking: bool = False
        self.callee_speaking_start_time: float = -1

        # Sequence management for response invalidation
        # -1 is reserved for background audio (backchanneling, ambient noise)
        self.curr_sequence_id: int = 0
        self.sequence_ids: Set[int] = {-1}

        # Turn tracking
        self.turn_id: int = 0

        # Timing state for incremental delay
        self.let_remaining_audio_pass_through: bool = False
        self.time_since_first_interim_result: float = -1
        self.required_delay_before_speaking: float = 0
        self.incremental_delay: int = incremental_delay

        # Configuration
        self.number_of_words_for_interruption: int = number_of_words_for_interruption
        self.accidental_interruption_phrases: Set[str] = set(
            accidental_interruption_phrases or []
        )

        logger.info(
            f"InterruptionManager initialized with: "
            f"number_of_words_for_interruption={number_of_words_for_interruption}, "
            f"incremental_delay={incremental_delay}ms"
        )

    # =========================================================================
    # Core Decision Methods
    # =========================================================================

    def should_send_audio(self, sequence_id: int) -> bool:
        """
        Decides if audio chunk should be sent to user.

        Part 1 (Extraction): Only checks sequence_id (matches current behavior).
        Part 2 (Bug Fix): Will add callee_speaking check.

        Args:
            sequence_id: The sequence ID of the audio chunk.

        Returns:
            True if the audio should be sent, False otherwise.
        """
        # Part 1: Match current behavior - only check sequence_id
        return sequence_id in self.sequence_ids

    def should_trigger_interruption(
        self,
        word_count: int,
        transcript: str,
        is_audio_playing: bool,
        welcome_played: bool,
    ) -> bool:
        """
        Decides if user speech should trigger an interruption.

        Args:
            word_count: Number of words in the interim transcript.
            transcript: The transcript content.
            is_audio_playing: Whether agent audio is currently playing.
            welcome_played: Whether the welcome message has been played.

        Returns:
            True if interruption should be triggered, False otherwise.
        """
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
        """
        Checks if a final transcript should be ignored as false interruption.

        This is used for speech_final events to filter out short utterances
        that don't qualify as intentional interruptions.

        Args:
            word_count: Number of words in the final transcript.
            transcript: The transcript content.
            is_audio_playing: Whether agent audio is currently playing.
            welcome_played: Whether the welcome message has been played.

        Returns:
            True if this should be ignored as false interruption, False otherwise.
        """
        if not is_audio_playing or not welcome_played:
            return False

        transcript_stripped = transcript.strip()
        return (
            word_count <= self.number_of_words_for_interruption
            and transcript_stripped not in self.accidental_interruption_phrases
        )

    # =========================================================================
    # State Management Methods
    # =========================================================================

    def on_user_speech_started(self) -> None:
        """
        Called when interim transcript received - user started speaking.

        Updates state to track that user is speaking and records timing
        for incremental delay logic.
        """
        if not self.callee_speaking:
            self.callee_speaking = True
            self.callee_speaking_start_time = time.time()
            logger.info(f"User started speaking at {self.callee_speaking_start_time}")

    def on_interim_transcript_received(self) -> None:
        """
        Called when any interim transcript is received.

        Updates timing and audio pass-through state.
        Should be called in addition to on_user_speech_started for each interim.
        """
        self.let_remaining_audio_pass_through = False

        if self.time_since_first_interim_result == -1:
            self.time_since_first_interim_result = time.time() * 1000
            logger.info(
                f"Setting time for first interim result as {self.time_since_first_interim_result}"
            )

    def on_user_speech_ended(self) -> None:
        """
        Called when speech_final/UtteranceEnd received.

        Resets speaking state and allows remaining audio to pass through.
        """
        self.callee_speaking = False
        self.let_remaining_audio_pass_through = True
        self.time_since_first_interim_result = -1
        logger.info("User speech ended")

    def on_interruption_triggered(self) -> None:
        """
        Called when cleanup is about to happen due to user interruption.

        Increments turn ID and invalidates pending responses.
        """
        self.turn_id += 1
        self.invalidate_pending_responses()
        logger.info(f"Interruption triggered, turn_id now {self.turn_id}")

    def invalidate_pending_responses(self) -> None:
        """
        Reset sequence_ids to invalidate all pending audio.

        After calling this, only background audio (sequence_id=-1) will be valid.
        """
        self.sequence_ids = {-1}
        logger.info("Pending responses invalidated, sequence_ids reset to {-1}")

    def get_next_sequence_id(self) -> int:
        """
        Generate and register a new sequence ID.

        Returns:
            The new sequence ID that was registered.
        """
        self.curr_sequence_id += 1
        self.sequence_ids.add(self.curr_sequence_id)
        return self.curr_sequence_id

    def reset_transmission_state(self) -> None:
        """
        Reset state after cleanup/interruption.

        Called after downstream tasks are cleaned up.
        """
        # Note: started_transmitting_audio stays in TaskManager
        # as it's more related to output handling
        pass

    # =========================================================================
    # Delay/Timing Methods
    # =========================================================================

    def should_delay_output(self, welcome_message_played: bool) -> Tuple[bool, float]:
        """
        Determines if output should be delayed and for how long.

        This implements the incremental delay logic that prevents the agent
        from responding too quickly to the user's first words.

        Args:
            welcome_message_played: Whether the welcome message has been played.

        Returns:
            Tuple of (should_delay, sleep_duration_seconds).
        """
        if not welcome_message_played:
            return False, 0

        if not self.let_remaining_audio_pass_through:
            return self._check_delay_without_passthrough()
        else:
            return self._check_delay_with_passthrough()

    def _check_delay_without_passthrough(self) -> Tuple[bool, float]:
        """Check delay when audio should NOT pass through."""
        if self.time_since_first_interim_result == -1:
            return True, 0.1

        elapsed = (time.time() * 1000) - self.time_since_first_interim_result
        if elapsed < self.required_delay_before_speaking:
            logger.info(
                f"Delay check: elapsed={elapsed}ms, required={self.required_delay_before_speaking}ms, sleeping 100ms"
            )
            return True, 0.1

        return False, 0

    def _check_delay_with_passthrough(self) -> Tuple[bool, float]:
        """Check delay when audio should pass through."""
        if self.time_since_first_interim_result == -1:
            return False, 0

        elapsed = (time.time() * 1000) - self.time_since_first_interim_result
        if elapsed < self.required_delay_before_speaking:
            logger.info(
                f"Delay check (passthrough): elapsed={elapsed}ms, required={self.required_delay_before_speaking}ms, sleeping 100ms"
            )
            return True, 0.1

        return False, 0

    def update_required_delay(self, history_length: int) -> None:
        """
        Update required delay based on conversation progress.

        After the first couple of turns, we add incremental delay to give
        the user more time before the agent responds.

        Args:
            history_length: Number of messages in conversation history.
        """
        if history_length > 2:
            self.required_delay_before_speaking = self.incremental_delay
        else:
            self.required_delay_before_speaking = 0
        logger.info(
            f"Updated required_delay_before_speaking to {self.required_delay_before_speaking}ms"
        )

    def reset_delay_for_speech_final(self, history_length: int, minimum_wait_duration: int) -> None:
        """
        Reset delay variables when speech_final is received.

        Args:
            history_length: Number of messages in conversation history.
            minimum_wait_duration: Minimum wait duration from config.
        """
        self.time_since_first_interim_result = -1
        if history_length > 2:
            self.required_delay_before_speaking = max(
                minimum_wait_duration - self.incremental_delay, 0
            )
        else:
            self.required_delay_before_speaking = 0

    # =========================================================================
    # Query Methods
    # =========================================================================

    def is_user_speaking(self) -> bool:
        """Check if user is currently speaking."""
        return self.callee_speaking

    def is_valid_sequence(self, sequence_id: int) -> bool:
        """
        Check if a sequence_id is still valid (not interrupted).

        Args:
            sequence_id: The sequence ID to check.

        Returns:
            True if the sequence ID is valid, False otherwise.
        """
        return sequence_id in self.sequence_ids

    def get_turn_id(self) -> int:
        """Get the current turn ID."""
        return self.turn_id

    def get_user_speaking_duration(self) -> float:
        """
        Get how long the user has been speaking.

        Returns:
            Duration in seconds, or 0 if user is not speaking.
        """
        if not self.callee_speaking or self.callee_speaking_start_time < 0:
            return 0
        return time.time() - self.callee_speaking_start_time

    def get_time_since_first_interim(self) -> float:
        """
        Get time elapsed since first interim result.

        Returns:
            Time in milliseconds, or -1 if no interim result received.
        """
        if self.time_since_first_interim_result == -1:
            return -1
        return (time.time() * 1000) - self.time_since_first_interim_result
