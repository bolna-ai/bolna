import numpy as np

FRAME_MS = 20
QUIET_RMS = 200
PAUSE_MS = 300
DEAD_AIR_MS = 600
MAX_PAUSES_PER_REPLY = 16


def _mulaw_table():
    u = ~np.arange(256, dtype=np.uint8)
    sign, exponent, mantissa = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    magnitude = ((mantissa.astype(np.int32) << 3) + 0x84) << exponent
    return np.where(sign != 0, 0x84 - magnitude, magnitude - 0x84).astype(np.int16)


_MULAW_TO_PCM = _mulaw_table()


def _samples(data, audio_format):
    """int16 samples of an outgoing audio chunk, or None for a format that can't be read sample by sample (mp3)."""
    if audio_format in ("mulaw", "ulaw"):
        return _MULAW_TO_PCM[np.frombuffer(data, dtype=np.uint8)]
    if audio_format in ("pcm", "wav"):
        if data[:4] == b"RIFF":
            start = data.find(b"data", 12)
            data = data[start + 8 :] if start != -1 else b""
        return np.frombuffer(data[: len(data) - len(data) % 2], dtype=np.int16)
    return None


class _Reply:
    __slots__ = ("frames", "carry", "quiet_frames", "spoke", "pauses", "unanalyzed_chunks")

    def __init__(self):
        self.frames = 0
        self.carry = np.empty(0, dtype=np.int16)
        self.quiet_frames = 0
        self.spoke = False
        self.pauses = []
        self.unanalyzed_chunks = 0


class TtsAudioMonitor:
    """Finds silences inside each agent reply in the audio sent to the caller, whatever the TTS provider."""

    def __init__(self, sampling_rate):
        self.frame_len = sampling_rate * FRAME_MS // 1000
        self.replies = {}

    def observe(self, meta_info, data):
        sequence_id = meta_info.get("sequence_id")
        if sequence_id is None or sequence_id < 0 or not isinstance(data, bytes) or len(data) <= 2:
            return
        reply = self.replies.setdefault(sequence_id, _Reply())
        samples = _samples(data, meta_info.get("format"))
        if samples is None:
            reply.unanalyzed_chunks += 1
            return
        if len(reply.carry):
            samples = np.concatenate((reply.carry, samples))
        frames = len(samples) // self.frame_len
        reply.carry = samples[frames * self.frame_len :]
        if frames:
            power = np.mean(samples[: frames * self.frame_len].astype(np.float32).reshape(frames, -1) ** 2, axis=1)
            self._track(reply, power < QUIET_RMS**2)
        if meta_info.get("end_of_synthesizer_stream"):
            # Silence after a stream ends, and before the next one starts, is not a pause in speech.
            reply.quiet_frames, reply.spoke, reply.carry = 0, False, reply.carry[:0]

    def _track(self, reply, quiet):
        starts = np.concatenate(([0], np.flatnonzero(np.diff(quiet.astype(np.int8))) + 1))
        lengths = np.diff(np.concatenate((starts, [len(quiet)])))
        for start, length, is_quiet in zip(starts.tolist(), lengths.tolist(), quiet[starts].tolist()):
            if is_quiet:
                if reply.spoke:
                    reply.quiet_frames += length
                continue
            if reply.quiet_frames * FRAME_MS >= PAUSE_MS:
                reply.pauses.append((reply.frames + start - reply.quiet_frames, reply.quiet_frames))
            reply.quiet_frames = 0
            reply.spoke = True
        reply.frames += len(quiet)

    def summary(self, chunk_marks):
        """One record per reply, counting only what the caller heard: a barge-in clears a reply's unplayed tail."""
        cleared_ms = {}
        for mark in chunk_marks:
            if mark.get("cleared_on_interrupt") and mark.get("sequence_id") is not None:
                cleared_ms[mark["sequence_id"]] = (
                    cleared_ms.get(mark["sequence_id"], 0) + (mark.get("duration") or 0) * 1000
                )
        replies = []
        for sequence_id, reply in sorted(self.replies.items()):
            audio_ms = reply.frames * FRAME_MS
            heard_ms = max(0, round(audio_ms - cleared_ms.get(sequence_id, 0)))
            pauses = [
                [offset * FRAME_MS, length * FRAME_MS]
                for offset, length in reply.pauses
                if (offset + length) * FRAME_MS <= heard_ms
            ]
            dead_air = [length for _, length in pauses if length >= DEAD_AIR_MS]
            replies.append(
                {
                    "sequence_id": sequence_id,
                    "audio_ms": audio_ms,
                    "heard_ms": heard_ms,
                    "dead_air_runs": len(dead_air),
                    "dead_air_max_ms": max(dead_air, default=0),
                    "dead_air_total_ms": sum(dead_air),
                    "pauses_ms": pauses[:MAX_PAUSES_PER_REPLY],
                    "unanalyzed_chunks": reply.unanalyzed_chunks,
                }
            )
        return {
            "replies": replies,
            "heard_ms": sum(r["heard_ms"] for r in replies),
            "dead_air_runs": sum(r["dead_air_runs"] for r in replies),
            "dead_air_ms": sum(r["dead_air_total_ms"] for r in replies),
        }
