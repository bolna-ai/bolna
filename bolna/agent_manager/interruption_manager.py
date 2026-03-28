import time
from typing import Dict, List, Optional, Set, Tuple
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

        # Live interruption counters
        self.user_interrupted_agent_count: int = 0
        self.agent_interrupted_user_count: int = 0
        self.barge_in_recovery_count: int = 0
        self._awaiting_recovery: bool = False

        # Per-event records; timestamps converted to call-relative ms in get_interruption_stats()
        self.interruption_events: List[Dict] = []
        self._open_event: Optional[Dict] = None  # latest event awaiting user_end_s

        # ASR turn IDs active at interruption time; used to annotate was_interrupted on turn_latencies
        self.interrupted_transcriber_turn_ids: Set = set()

        # Speaking time accumulators (talk-to-listen ratio)
        self._total_user_speaking_ms: float = 0.0
        self._total_agent_speaking_ms: float = 0.0
        self._agent_speaking_start_time: float = -1  # -1 when agent not speaking
        self._last_sent_sequence_id: Optional[int] = None  # dedup guard for on_agent_speech_started
        self.longest_agent_monologue_ms: float = 0.0

        logger.info(
            f"InterruptionManager initialized: "
            f"words_for_interruption={number_of_words_for_interruption}, "
            f"incremental_delay={incremental_delay}ms"
        )

    # ── Audio gate ────────────────────────────────────────────────────────────

    def get_audio_send_status(self, sequence_id: int, history_length: int = 0) -> str:
        """
        Centralized decision for whether audio should be sent.

        Returns:
        - "SEND" - audio should be sent now
        - "BLOCK" - audio should be discarded (invalid/cancelled sequence)
        - "WAIT" - audio should be delayed (grace period active)
        """
        # Check 1: Invalid sequence - discard
        if sequence_id not in self.sequence_ids:
            return "BLOCK"

        # Check 2: User is speaking - hold audio until they stop
        # Only invalid sequences (from real interruptions) get hard BLOCK above
        if self.callee_speaking:
            logger.info(f"Audio status=WAIT - user is speaking")
            return "WAIT"

        # Check 3: Grace period (only after first 2 turns to avoid latency on welcome)
        if history_length > 2:
            time_since_utterance_end = self.get_time_since_utterance_end()
            if time_since_utterance_end != -1 and time_since_utterance_end < self.incremental_delay:
                logger.info(f"Audio status=WAIT - grace period: {time_since_utterance_end:.0f}ms / {self.incremental_delay}ms")
                return "WAIT"

        return "SEND"

    # ── Interruption gate ─────────────────────────────────────────────────────

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

    # ── Speech lifecycle callbacks ────────────────────────────────────────────

    def on_user_speech_started(self) -> None:
        """Called when user starts speaking (first interim transcript received)."""
        if not self.callee_speaking:
            self.callee_speaking = True
            self.callee_speaking_start_time = time.time()
            logger.info("User started speaking")

    def on_interim_transcript_received(self) -> None:
        """Called on each interim transcript to update timing state."""
        self.let_remaining_audio_pass_through = False

        if self.time_since_first_interim_result == -1:
            self.time_since_first_interim_result = time.time() * 1000
            logger.info(f"First interim at {self.time_since_first_interim_result}")

    def on_user_speech_ended(self, update_utterance_time: bool = True) -> None:
        """Called when user stops speaking (speech_final / UtteranceEnd).

        update_utterance_time=False keeps the grace period anchored to the last
        real turn (use for false interruptions or late UtteranceEnd events).
        """
        self.callee_speaking = False
        self.let_remaining_audio_pass_through = True
        self.time_since_first_interim_result = -1

        now_s = time.time()
        if update_utterance_time:
            self.utterance_end_time = now_s * 1000

        if self.callee_speaking_start_time > 0:
            self._total_user_speaking_ms += (now_s - self.callee_speaking_start_time) * 1000

        if self._open_event is not None and self._open_event.get("user_end_s") is None:
            self._open_event["user_end_s"] = now_s
            self._open_event = None

        logger.info("User speech ended")

    # ── Interruption event callbacks ──────────────────────────────────────────

    def on_interruption_triggered(self) -> None:
        """User barged in while agent was speaking. user_end_s filled later by on_user_speech_ended()."""
        self.turn_id += 1
        self.user_interrupted_agent_count += 1
        self._awaiting_recovery = True
        self._finalize_agent_speaking_session()
        self.invalidate_pending_responses()

        event: Dict = {
            "type": "user_interrupted_agent",
            "user_start_s": self.callee_speaking_start_time if self.callee_speaking_start_time > 0 else None,
            "user_end_s": None,
            "recovery_completed": False,
        }
        self.interruption_events.append(event)
        self._open_event = event

        logger.info(f"Interruption triggered — turn_id={self.turn_id}, user_interrupted_agent_count={self.user_interrupted_agent_count}")

    def on_agent_interrupted_user(self) -> None:
        """Agent responded prematurely and was cancelled within the grace period.
        Does NOT set _awaiting_recovery — recovery is only tracked for user barge-ins
        to keep barge_in_recovery_rate denominator consistent with user_interrupted_agent_count.
        user_end_s left open; filled by on_user_speech_ended() when user finishes speaking.
        """
        self.agent_interrupted_user_count += 1

        event: Dict = {
            "type": "agent_interrupted_user",
            "user_start_s": self.callee_speaking_start_time if self.callee_speaking_start_time > 0 else None,
            "user_end_s": None,
            "recovery_completed": False,
        }
        self.interruption_events.append(event)
        self._open_event = event

        logger.info(f"Agent interrupted user — agent_interrupted_user_count={self.agent_interrupted_user_count}")

    def record_interrupted_transcriber_turn(self, turn_id) -> None:
        """Store ASR turn_id at interruption time for was_interrupted annotation at call end."""
        if turn_id is not None:
            self.interrupted_transcriber_turn_ids.add(turn_id)
            logger.info(f"Recorded interrupted transcriber turn_id={turn_id}")

    # ── Agent speaking lifecycle ──────────────────────────────────────────────

    def on_agent_speech_started(self, sequence_id: int) -> None:
        """First audio chunk entering SEND path. Deduplicated by sequence_id."""
        if sequence_id == self._last_sent_sequence_id:
            return
        self._last_sent_sequence_id = sequence_id
        self._agent_speaking_start_time = time.time()
        logger.info(f"Agent speech started (sequence_id={sequence_id})")

    def on_agent_speech_ended(self) -> None:
        """Clean end of agent audio (end_of_synthesizer_stream in SEND path)."""
        self._finalize_agent_speaking_session()

    def _finalize_agent_speaking_session(self) -> None:
        """Close current agent speaking window; called on clean end or barge-in."""
        if self._agent_speaking_start_time <= 0:
            return
        duration_ms = (time.time() - self._agent_speaking_start_time) * 1000
        self._total_agent_speaking_ms += duration_ms
        if duration_ms > self.longest_agent_monologue_ms:
            self.longest_agent_monologue_ms = duration_ms
        self._agent_speaking_start_time = -1
        logger.info(f"Agent speaking session closed: {duration_ms:.0f}ms")

    def on_successful_response_delivered(self) -> None:
        """Agent delivered a full response after an interruption — counts as recovery."""
        if not self._awaiting_recovery:
            return

        self.barge_in_recovery_count += 1
        self._awaiting_recovery = False

        for event in reversed(self.interruption_events):
            if not event["recovery_completed"]:
                event["recovery_completed"] = True
                break

        logger.info(f"Barge-in recovery confirmed — barge_in_recovery_count={self.barge_in_recovery_count}")

    # ── Stats output ──────────────────────────────────────────────────────────

    def get_interruption_stats(self, call_start_ms: float) -> Dict:
        """Return call-level interruption stats for latency_dict.
        Timestamps converted from epoch seconds to ms relative to call_start_ms.
        """
        total = self.user_interrupted_agent_count + self.agent_interrupted_user_count

        recovery_rate: Optional[float] = None
        if self.user_interrupted_agent_count > 0:
            recovery_rate = round(
                self.barge_in_recovery_count / self.user_interrupted_agent_count * 100, 1
            )

        events = []
        for e in self.interruption_events:
            entry: Dict = {
                "type": e["type"],
                "recovery_completed": e["recovery_completed"],
            }
            if e.get("user_start_s") is not None:
                entry["user_start_ms"] = round(e["user_start_s"] * 1000 - call_start_ms, 2)
            if e.get("user_end_s") is not None:
                entry["user_end_ms"] = round(e["user_end_s"] * 1000 - call_start_ms, 2)
            if "user_start_ms" in entry and "user_end_ms" in entry:
                entry["user_duration_ms"] = round(entry["user_end_ms"] - entry["user_start_ms"], 2)
            events.append(entry)

        # Snapshot agent_ms including any still-open session
        agent_ms = self._total_agent_speaking_ms
        if self._agent_speaking_start_time > 0:
            agent_ms += (time.time() - self._agent_speaking_start_time) * 1000

        user_ms = self._total_user_speaking_ms
        total_speaking_ms = agent_ms + user_ms

        talk_to_listen_ratio: Optional[float] = None
        if total_speaking_ms > 0:
            talk_to_listen_ratio = round(agent_ms / total_speaking_ms * 100, 1)

        return {
            "user_interrupted_agent_count": self.user_interrupted_agent_count,
            "agent_interrupted_user_count": self.agent_interrupted_user_count,
            "total_interruptions": total,
            "barge_in_recovery_count": self.barge_in_recovery_count,
            "barge_in_recovery_rate": recovery_rate,  # Hamming benchmark: >90% good, <80% critical
            "interruption_events": events,
            "agent_speaking_ms": round(agent_ms),
            "user_speaking_ms": round(user_ms),
            "talk_to_listen_ratio": talk_to_listen_ratio,
            "longest_agent_monologue_ms": round(self.longest_agent_monologue_ms),
        }

    # ── Sequence management ───────────────────────────────────────────────────

    def invalidate_pending_responses(self) -> None:
        """Invalidates all pending audio by resetting sequence_ids."""
        self.sequence_ids = {-1}
        logger.info("Pending responses invalidated")

    def revalidate_sequence_id(self, sequence_id: int) -> None:
        """Re-adds a sequence ID after invalidation when a new response's
        sequence_id was already allocated by __get_updated_meta_info."""
        self.sequence_ids.add(sequence_id)
        logger.info(f"Re-validated sequence_id={sequence_id}")

    def get_next_sequence_id(self) -> int:
        """Generates and registers a new sequence ID."""
        self.curr_sequence_id += 1
        self.sequence_ids.add(self.curr_sequence_id)
        return self.curr_sequence_id

    # ── Delay logic ───────────────────────────────────────────────────────────

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

    # ── State queries ─────────────────────────────────────────────────────────

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
