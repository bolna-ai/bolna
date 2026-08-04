import asyncio
import copy
import os
import time
import zlib
from collections import OrderedDict
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from bolna.constants import IS_USER_ONLINE_MESSAGE
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

HIGH_DELAY_THRESHOLD = 2.0

# Per-chunk mark records are kept for the whole call so post-call analysis can see what was
# actually played, and nothing prunes them mid-call. The bound is on record count, not on age
# or call duration, because repetition depth drives the count: an agent stuck re-sending the
# same response grows this dict for as long as the call lasts, while a long but well-behaved
# call stays flat. 1000 records covers roughly fifteen minutes of continuous agent speech.
MAX_MARK_HISTORY = int(os.getenv("MAX_MARK_HISTORY", "1000"))

# Raw chunk marks are ~90% of a latency_dict payload (11x the transcript), so they are sampled
# per call rather than stored for every one. The mark_tracking aggregate is always kept.
PERSIST_CHUNK_MARKS_PCT = int(os.getenv("PERSIST_CHUNK_MARKS_PCT", "0"))

# Interrupts dump the pending mark ids; a stuck agent can leave hundreds pending.
CLEAR_LOG_MARK_ID_LIMIT = 20


def should_persist_chunk_marks(run_id: Optional[str]) -> bool:
    """Whether this call's raw chunk marks should be persisted alongside the aggregate.

    Bucketed on run_id so the decision is stable for a call and the sampled set is spread
    evenly across traffic rather than clustered in time. crc32 is the bucketing function
    because this is a checksum over an opaque id, not a digest anything relies on.
    """
    if PERSIST_CHUNK_MARKS_PCT <= 0:
        return False
    if PERSIST_CHUNK_MARKS_PCT >= 100:
        return True
    if not run_id:
        return False
    return zlib.crc32(str(run_id).encode()) % 100 < PERSIST_CHUNK_MARKS_PCT


class SequenceStats(BaseModel):
    sent: int = 0
    acked: int = 0
    delays: List[float] = Field(default_factory=list)
    interrupted: bool = False
    first_sent_ts: Optional[float] = None
    last_sent_ts: Optional[float] = None
    total_audio_duration: float = 0
    turn_id: Optional[int] = None


class SequenceSummary(BaseModel):
    seq: int
    sent: int
    acked: int
    max_delay: float = 0
    avg_delay: float = 0
    interrupted: bool = False
    chunk_delays: List[float] = Field(default_factory=list)
    tts_speed_ratio: Optional[float] = None
    audio_duration_s: float = 0


class MarkTrackingSummary(BaseModel):
    total_sent: int = 0
    total_acked: int = 0
    total_missed: int = 0
    max_delay_s: float = 0
    avg_delay_s: float = 0
    high_delay_count: int = 0
    per_sequence: List[SequenceSummary] = Field(default_factory=list)
    # Send time of the call's first audio chunk. Kept here because it survives history
    # eviction and is the calibration anchor recording analysis needs.
    first_mark_sent_ts: Optional[float] = None
    # Chunk-mark records still held vs evicted by the history cap, so a short raw mark list
    # is distinguishable from a quiet call.
    history_retained: int = 0
    history_dropped: int = 0


class MarkStats(BaseModel):
    total_sent: int = 0
    total_acked: int = 0
    delays: List[float] = Field(default_factory=list)
    per_sequence: Dict[int, SequenceStats] = Field(default_factory=dict)

    def ensure_sequence(self, sequence_id: int):
        if sequence_id not in self.per_sequence:
            self.per_sequence[sequence_id] = SequenceStats()


class MarkEventMetaData:
    def __init__(self, max_history: Optional[int] = None):
        self.mark_event_meta_data = {}
        self.previous_mark_event_meta_data = {}
        self._mark_history: "OrderedDict[str, Dict]" = OrderedDict()
        self._max_history = MAX_MARK_HISTORY if max_history is None else max_history
        self._history_dropped = 0
        self._first_mark_sent_ts: Optional[float] = None
        self.counter = 0
        self.mark_changed = asyncio.Event()
        self._mark_stats = MarkStats()
        self.heard_text_by_turn = {}
        self.heard_text_by_response = {}
        self.last_heard_turn_id = None
        self.last_heard_response_uid = None
        self.welcome_pre_mark_id = None
        self.audio_playing_until = 0.0

    def _note_audio_queued(self, value, duration):
        # Audio is handed over faster than real time, so a chunk starts playing when the
        # previously queued audio ends, not when it is sent. The online-check prompt is left out
        # so it does not postpone the inactivity hangup, as in final_chunk_played_observable.
        if duration <= 0 or value.get("type") == IS_USER_ONLINE_MESSAGE:
            return
        self.audio_playing_until = max(self.audio_playing_until, time.time()) + duration

    def get_audio_playing_until(self) -> float:
        """Estimated wall-clock end of queued agent audio; in the past once playback is done."""
        return self.audio_playing_until

    def drop_playout_estimate(self) -> None:
        """Queued audio was discarded, so nothing is playing."""
        self.audio_playing_until = 0.0

    def update_data(self, mark_id, value):
        value["counter"] = self.counter
        value.setdefault("acked", False)
        value.setdefault("ack_ts", None)
        self.counter += 1
        self.mark_event_meta_data[mark_id] = value
        duration = value.get("duration") or 0
        if value.get("type") != "pre_mark_message":
            self._record_history(mark_id, value)
        logger.info(
            "BOLNA_TRACE_MARK update mark_id=%s type=%s seq=%s turn=%s response_uid=%s group_uid=%s counter=%s dur=%.3f text_len=%s",
            mark_id,
            value.get("type"),
            value.get("sequence_id"),
            value.get("turn_id"),
            value.get("response_uid"),
            value.get("response_group_uid"),
            value.get("counter"),
            duration,
            len(value.get("text_synthesized", "") or ""),
        )
        self.mark_changed.set()
        if value.get("type") != "pre_mark_message":
            self._note_audio_queued(value, duration)
            self._mark_stats.total_sent += 1
            seq = value.get("sequence_id")
            if seq is not None:
                self._mark_stats.ensure_sequence(seq)
                entry = self._mark_stats.per_sequence[seq]
                entry.sent += 1

                now = time.time()
                if entry.first_sent_ts is None:
                    entry.first_sent_ts = now
                entry.last_sent_ts = now
                if duration > 0:
                    entry.total_audio_duration += duration
                turn_id = value.get("turn_id")
                if turn_id is not None and entry.turn_id is None:
                    entry.turn_id = turn_id

    def _record_history(self, mark_id, value):
        """Append to the bounded per-chunk history, evicting the oldest record when full.

        Records are stored by reference, so a later ack or interrupt flag written onto the same
        dict stays visible here. Eviction drops only that reference — the live
        mark_event_meta_data entry and the mark_tracking aggregate are unaffected, which is why
        capping here costs nothing but detail on the pathological calls that hit the cap.
        """
        if self._first_mark_sent_ts is None and value.get("sent_ts"):
            self._first_mark_sent_ts = value["sent_ts"]

        self._mark_history[mark_id] = value
        if len(self._mark_history) <= self._max_history:
            return

        while len(self._mark_history) > self._max_history:
            self._mark_history.popitem(last=False)
            self._history_dropped += 1
        if self._history_dropped == 1:
            # Once per call: past this point the raw mark list no longer covers the whole call.
            logger.warning("mark history hit its %s-record cap, dropping oldest chunk marks", self._max_history)

    def release_call_buffers(self):
        """Free the post-call mark buffers once the call output has been snapshotted.

        Every reader runs during the call or at snapshot time — get_chunk_marks and
        get_mark_tracking_summary build the output, get_heard_text_for_* and
        fetch_cleared_mark_event_data serve the interruption path. Teardown after the snapshot
        (recording upload to S3, metrics, DB writes) runs for seconds with the TaskManager still
        referenced, so holding these until it is collected keeps the pod's memory floor up for
        no benefit.
        """
        self._mark_history.clear()
        self.heard_text_by_turn.clear()
        self.heard_text_by_response.clear()
        self.previous_mark_event_meta_data = {}

    def record_ack(self, delay, sequence_id):
        self._mark_stats.total_acked += 1
        if delay >= 0:
            self._mark_stats.delays.append(delay)
        if sequence_id is not None:
            self._mark_stats.ensure_sequence(sequence_id)
            entry = self._mark_stats.per_sequence[sequence_id]
            entry.acked += 1
            if delay >= 0:
                entry.delays.append(delay)

    def record_heard_text(self, mark_data, heard_text):
        if not heard_text:
            return

        turn_id = mark_data.get("turn_id")
        if turn_id is not None:
            self.last_heard_turn_id = turn_id
            self.heard_text_by_turn[turn_id] = self.heard_text_by_turn.get(turn_id, "") + heard_text

        response_uid = mark_data.get("response_uid")
        if response_uid is not None:
            self.last_heard_response_uid = response_uid
            self.heard_text_by_response[response_uid] = self.heard_text_by_response.get(response_uid, "") + heard_text

    def get_heard_text_for_turn(self, turn_id=None):
        if turn_id is None:
            turn_id = self.last_heard_turn_id
        if turn_id is None:
            return ""
        return (self.heard_text_by_turn.get(turn_id) or "").strip()

    def get_heard_text_for_response(self, response_uid=None):
        if response_uid is None:
            response_uid = self.last_heard_response_uid
        if response_uid is None:
            return ""
        return (self.heard_text_by_response.get(response_uid) or "").strip()

    def fetch_data(self, mark_id):
        entry = self.mark_event_meta_data.get(mark_id)
        if entry is not None and entry.get("type") != "pre_mark_message":
            entry["acked"] = True
            entry["ack_ts"] = time.time()
        result = self.mark_event_meta_data.pop(mark_id, {})
        if result:
            logger.info(
                "BOLNA_TRACE_MARK fetch mark_id=%s type=%s seq=%s turn=%s response_uid=%s group_uid=%s counter=%s",
                mark_id,
                result.get("type"),
                result.get("sequence_id"),
                result.get("turn_id"),
                result.get("response_uid"),
                result.get("response_group_uid"),
                result.get("counter"),
            )
            self.mark_changed.set()
        return result

    def clear_data(self):
        logger.info(f"Clearing mark meta data dict")
        pending_ids = list(self.mark_event_meta_data.keys())
        logger.info(
            "BOLNA_TRACE_MARK clear pending=%s mark_ids=%s",
            len(pending_ids),
            pending_ids[:CLEAR_LOG_MARK_ID_LIMIT],
        )
        self.counter = 0
        self.drop_playout_estimate()

        for mark_id, value in self.mark_event_meta_data.items():
            if value.get("type") != "pre_mark_message":
                value["cleared_on_interrupt"] = True
                seq = value.get("sequence_id")
                if seq is not None and seq in self._mark_stats.per_sequence:
                    self._mark_stats.per_sequence[seq].interrupted = True

        self.previous_mark_event_meta_data = copy.deepcopy(self.mark_event_meta_data)
        self.mark_event_meta_data = {}
        self.mark_changed.set()

    def get_mark_tracking_summary(self) -> dict:
        stats = self._mark_stats
        all_delays = stats.delays

        summary = MarkTrackingSummary(
            total_sent=stats.total_sent,
            total_acked=stats.total_acked,
            total_missed=stats.total_sent - stats.total_acked,
            max_delay_s=round(max(all_delays), 3) if all_delays else 0,
            avg_delay_s=round(sum(all_delays) / len(all_delays), 3) if all_delays else 0,
            high_delay_count=sum(1 for d in all_delays if d > HIGH_DELAY_THRESHOLD),
            first_mark_sent_ts=self._first_mark_sent_ts,
            history_retained=len(self._mark_history),
            history_dropped=self._history_dropped,
        )

        for seq_id in sorted(stats.per_sequence.keys()):
            seq = stats.per_sequence[seq_id]

            wall_clock = 0
            if seq.first_sent_ts and seq.last_sent_ts:
                wall_clock = seq.last_sent_ts - seq.first_sent_ts
            tts_speed_ratio = round(seq.total_audio_duration / wall_clock, 2) if wall_clock > 0 else None

            summary.per_sequence.append(
                SequenceSummary(
                    seq=seq_id,
                    sent=seq.sent,
                    acked=seq.acked,
                    max_delay=round(max(seq.delays), 3) if seq.delays else 0,
                    avg_delay=round(sum(seq.delays) / len(seq.delays), 3) if seq.delays else 0,
                    interrupted=seq.interrupted,
                    chunk_delays=[round(d, 3) for d in seq.delays],
                    tts_speed_ratio=tts_speed_ratio,
                    audio_duration_s=round(seq.total_audio_duration, 3),
                )
            )

        return summary.model_dump()

    def fetch_cleared_mark_event_data(self):
        return self.previous_mark_event_meta_data

    def get_chunk_marks(self) -> List[Dict]:
        """Per-mark wall-clock detail for post-call audio analysis.

        Returns one dict per agent audio chunk (excluding pre_mark_message), ordered
        by send counter. Each entry carries the actually-spoken text, send/ack
        timestamps, sequence/turn linkage, and an interruption flag.

        Covers at most the last MAX_MARK_HISTORY chunks; get_mark_tracking_summary reports
        how many were evicted, and the call's first send timestamp survives eviction there.
        """
        out = []
        for mark_id, data in self._mark_history.items():
            if data.get("type") == "pre_mark_message":
                continue
            out.append(
                {
                    "mark_id": mark_id,
                    "sequence_id": data.get("sequence_id"),
                    "type": data.get("type"),
                    "text_synthesized": data.get("text_synthesized", ""),
                    "is_first_chunk": data.get("is_first_chunk", False),
                    "is_final_chunk": data.get("is_final_chunk", False),
                    "sent_ts": data.get("sent_ts"),
                    "duration": data.get("duration"),
                    "counter": data.get("counter"),
                    "acked": data.get("acked", False),
                    "ack_ts": data.get("ack_ts"),
                    "cleared_on_interrupt": data.get("cleared_on_interrupt", False),
                }
            )
        out.sort(key=lambda m: m["sent_ts"] if m.get("sent_ts") is not None else 0)
        return out

    def __str__(self):
        return f"{self.mark_event_meta_data}"
