"""Shaping for speech servers that render one request per piece of text.

Such a server ends every request the way a sentence ends and pads it with silence, so where a reply is cut and how
much silence sits at each join decide whether the reply sounds like one utterance.
"""

import re

import numpy as np

SENTENCE_PAUSE_MS = 300
CLAUSE_PAUSE_MS = 120
FINAL_TAIL_MS = 100
# The longest silence left inside a request: the model's own pauses run up to 0.8 s, mid-phrase as well.
MAX_PAUSE_MS = 300
LEAD_MS = 20
# Text held back for the next piece waits on the LLM; past this the audio already sent may run out first.
MAX_HELD_CHARACTERS = 100

_CLOSERS = r"[\"'”’)\]]*"
_SENTENCE_END = re.compile(rf"[.?!।॥]+{_CLOSERS}(?=\s|$)")
_CLAUSE_END = re.compile(rf"[,;:—–]{_CLOSERS}(?=\s)")
_ABBREVIATION = re.compile(r"(?:\b[A-Za-z]\.|\b(?:Rs|Mr|Mrs|Ms|Dr|No|St|vs|etc)\.)$")

_FRAME_MS = 10
_SILENCE_DBFS = -50.0
_PAUSE_FRAMES = 20
_SPEECH_FRAMES = 8


def _last_end(pattern, text, start=0):
    end = None
    for match in pattern.finditer(text, start):
        if match.group().startswith(".") and (
            _ABBREVIATION.search(text[: match.end()]) or text[match.end() :].lstrip()[:1].isdigit()
        ):
            continue
        end = match.end()
    return end


def split_speakable(text):
    """(text to speak now, text to hold for the next piece, pause after the spoken part in ms).

    Speaks up to the last sentence end, or the last clause end when that would hold back too much or there is no
    sentence end; text with neither is spoken whole with no pause, since the phrase continues in the next piece.
    """
    cut, pause = _last_end(_SENTENCE_END, text), SENTENCE_PAUSE_MS
    if cut is None or len(text) - cut > MAX_HELD_CHARACTERS:
        clause = _last_end(_CLAUSE_END, text, cut or 0)
        if clause is not None:
            cut, pause = clause, CLAUSE_PAUSE_MS
    if cut is None:
        return text, "", 0
    return text[:cut].rstrip(), text[cut:].strip(), pause


class PcmEdgeTrimmer:
    """Streams 16-bit mono PCM through, replacing a request's leading and trailing silence with a set length.

    Audio is passed on as it arrives. Only silence is held back, until speech follows it (then it was a pause inside
    the request and is passed on, shortened to MAX_PAUSE_MS) or the request ends (then it is the tail). A burst shorter than real speech
    after a pause is held with the silence, so a click at the end of a request goes with the tail. The first sound
    of a request is passed on at once, so first audio never waits.
    """

    def __init__(self, sample_rate):
        self.frame_bytes = sample_rate * _FRAME_MS // 1000 * 2
        self.reset()

    def reset(self):
        self._carry = b""
        self._held = bytearray()
        self._silent_run = 0
        self._paused = False
        self._burst = 0
        self._speech_at = 0
        self._started = False

    def _ms(self, milliseconds):
        return milliseconds // _FRAME_MS * self.frame_bytes

    def _is_silent(self, frame):
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples)))
        return 20 * np.log10(rms / 32768 + 1e-9) < _SILENCE_DBFS

    def _release(self):
        held = bytes(self._held)
        if not self._started:
            held = held[max(0, self._speech_at - self._ms(LEAD_MS)) :]
            self._started = True
        elif self._speech_at > self._ms(MAX_PAUSE_MS):
            # Keeps the decay into the pause and the onset out of it.
            half = self._ms(MAX_PAUSE_MS) // 2
            held = held[:half] + held[self._speech_at - half :]
        self._held.clear()
        self._silent_run, self._paused, self._burst = 0, False, 0
        return held

    def feed(self, chunk):
        data = self._carry + chunk
        whole = len(data) - len(data) % self.frame_bytes
        data, self._carry = data[:whole], data[whole:]
        out = bytearray()
        for start in range(0, len(data), self.frame_bytes):
            frame = data[start : start + self.frame_bytes]
            self._held += frame
            if self._is_silent(frame):
                self._silent_run += 1
                self._paused = self._paused or (self._started and self._silent_run >= _PAUSE_FRAMES)
                continue
            self._silent_run = 0
            if not self._burst:
                self._speech_at = len(self._held) - self.frame_bytes
            if not self._paused:
                out += self._release()
                continue
            self._burst += 1
            if self._burst >= _SPEECH_FRAMES:
                out += self._release()
        return bytes(out)

    def finish(self, tail_ms, pad=False):
        """The request's tail: up to tail_ms of its trailing silence, padded to tail_ms with silence when pad is set."""
        keep = self._ms(tail_ms)
        held = bytes(self._held) if self._started else b""
        tail = bytes(len(held[:keep])) if self._paused else held[:keep]
        if pad:
            tail += bytes(keep - len(tail))
        self.reset()
        return tail
