import asyncio
import copy
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

HIGH_DELAY_THRESHOLD = 2.0


class SequenceStats(BaseModel):
    sent: int = 0
    acked: int = 0
    delays: List[float] = Field(default_factory=list)
    interrupted: bool = False
    first_sent_ts: Optional[float] = None
    last_sent_ts: Optional[float] = None
    total_audio_duration: float = 0


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


class MarkStats(BaseModel):
    total_sent: int = 0
    total_acked: int = 0
    delays: List[float] = Field(default_factory=list)
    per_sequence: Dict[int, SequenceStats] = Field(default_factory=dict)

    def ensure_sequence(self, sequence_id: int):
        if sequence_id not in self.per_sequence:
            self.per_sequence[sequence_id] = SequenceStats()


class MarkEventMetaData:
    def __init__(self):
        self.mark_event_meta_data = {}
        self.previous_mark_event_meta_data = {}
        self.counter = 0
        self.mark_changed = asyncio.Event()
        self._mark_stats = MarkStats()

    def update_data(self, mark_id, value):
        value['counter'] = self.counter
        self.counter += 1
        self.mark_event_meta_data[mark_id] = value

        if value.get("type") != "pre_mark_message":
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
                duration = value.get("duration", 0)
                if duration > 0:
                    entry.total_audio_duration += duration

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

    def fetch_data(self, mark_id):
        result = self.mark_event_meta_data.pop(mark_id, {})
        if result:
            self.mark_changed.set()
        return result

    def clear_data(self):
        logger.info(f"Clearing mark meta data dict")
        self.counter = 0

        for mark_id, value in self.mark_event_meta_data.items():
            if value.get("type") != "pre_mark_message":
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
        )

        for seq_id in sorted(stats.per_sequence.keys()):
            seq = stats.per_sequence[seq_id]

            wall_clock = 0
            if seq.first_sent_ts and seq.last_sent_ts:
                wall_clock = seq.last_sent_ts - seq.first_sent_ts
            tts_speed_ratio = round(seq.total_audio_duration / wall_clock, 2) if wall_clock > 0 else None

            summary.per_sequence.append(SequenceSummary(
                seq=seq_id,
                sent=seq.sent,
                acked=seq.acked,
                max_delay=round(max(seq.delays), 3) if seq.delays else 0,
                avg_delay=round(sum(seq.delays) / len(seq.delays), 3) if seq.delays else 0,
                interrupted=seq.interrupted,
                chunk_delays=[round(d, 3) for d in seq.delays],
                tts_speed_ratio=tts_speed_ratio,
                audio_duration_s=round(seq.total_audio_duration, 3),
            ))

        return summary.model_dump()

    def fetch_cleared_mark_event_data(self):
        return self.previous_mark_event_meta_data

    def __str__(self):
        return f"{self.mark_event_meta_data}"
