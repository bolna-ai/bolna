"""Shaping for speech servers that render one request per piece of text.

Such a server ends every request the way a sentence ends and pads it with silence, so where a reply is cut and how
much silence sits at each join decide whether the reply sounds like one utterance.
"""

import re

import numpy as np

SENTENCE_PAUSE_MS = 300
CLAUSE_PAUSE_MS = 120
FINAL_TAIL_MS = 100
LEAD_MS = 20
# Text held back for the next piece waits on the LLM; past this the audio already sent may run out first, and a
# reply's first audio would wait too long.
MAX_HELD_CHARACTERS = 100

_CLOSERS = r"[\"'”’)\]]*"
_SENTENCE_END = re.compile(rf"[.?!।॥]+{_CLOSERS}(?=\s|$)")
_CLAUSE_END = re.compile(rf"[,;:—–]{_CLOSERS}(?=\s)")
_ABBREVIATION = re.compile(r"(?:\b[A-Za-z]\.|\b(?:Rs|Mr|Mrs|Ms|Dr|No|St|vs|etc)\.)$")

_FRAME_MS = 10
_SILENCE_DBFS = -50.0
# A frame is silent when its mean square is below this: _SILENCE_DBFS of full scale, as power.
_SILENCE_POWER = (32768 * (10 ** (_SILENCE_DBFS / 20) - 1e-9)) ** 2


def _last_end(pattern, text, start=0):
    end = None
    for match in pattern.finditer(text, start):
        if match.group().startswith(".") and (
            _ABBREVIATION.search(text[: match.end()]) or text[match.end() :].lstrip()[:1].isdigit()
        ):
            continue
        # Servers of this kind misread a code or number that ends a request ("BS100," came out as "BS").
        if pattern is _CLAUSE_END:
            before = text[: match.start()].split()[-1:]
            if before and any(ch.isdigit() for ch in before[0]):
                continue
        end = match.end()
    return end


def split_speakable(text):
    """(text to speak now, text to hold for the next piece, pause after the spoken part in ms).

    Speaks up to the last sentence end. Text with no sentence end waits for the rest of its sentence; once more
    than MAX_HELD_CHARACTERS would wait, it is spoken up to its last clause end, or whole when it has none.
    """
    sentence = _last_end(_SENTENCE_END, text)
    if len(text) - (sentence or 0) > MAX_HELD_CHARACTERS:
        clause = _last_end(_CLAUSE_END, text, sentence or 0)
        if clause is not None:
            return text[:clause].rstrip(), text[clause:].strip(), CLAUSE_PAUSE_MS
        if sentence is None:
            return text, "", 0
    if sentence is None:
        return "", text, 0
    return text[:sentence].rstrip(), text[sentence:].strip(), SENTENCE_PAUSE_MS


class PcmEdgeTrimmer:
    """Streams one request's 16-bit mono PCM through, replacing its leading and trailing silence with a set length.

    Audio is passed on as it arrives. Once speech has begun, the first pass_ms of each silence is passed on too; the
    rest is held back until sound follows it (then it is passed on whole) or the request ends (then it is the tail).
    Nothing but silence is ever removed.
    """

    def __init__(self, sample_rate, pass_ms=0):
        self.frame_bytes = sample_rate * _FRAME_MS // 1000 * 2
        self._carry = b""
        self._held = bytearray()
        self._started = False
        self._pass = self._ms(pass_ms)
        # Bytes of the current silence already passed on.
        self._passed = 0

    def _ms(self, milliseconds):
        return milliseconds // _FRAME_MS * self.frame_bytes

    def feed(self, chunk):
        data = self._carry + chunk
        whole = len(data) - len(data) % self.frame_bytes
        data, self._carry = data[:whole], data[whole:]
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32).reshape(-1, self.frame_bytes // 2)
        silent = (samples * samples).mean(axis=1) < _SILENCE_POWER
        out = bytearray()
        for index, is_silent in enumerate(silent):
            frame = data[index * self.frame_bytes : (index + 1) * self.frame_bytes]
            if is_silent:
                if self._started and self._passed < self._pass:
                    out += frame
                    self._passed += len(frame)
                else:
                    self._held += frame
                continue
            out += self._held if self._started else self._held[max(0, len(self._held) - self._ms(LEAD_MS)) :]
            out += frame
            self._held.clear()
            self._passed = 0
            self._started = True
        return bytes(out)

    @property
    def passed_ms(self):
        """Milliseconds of the current silence already passed on; just before finish, the part of the tail sent."""
        return self._passed // self.frame_bytes * _FRAME_MS

    def finish(self, tail_ms, pad=False):
        """The request's tail: up to tail_ms of its trailing silence, padded to tail_ms with silence when pad is set."""
        keep = max(0, self._ms(tail_ms) - self._passed)
        tail = bytes(self._held[:keep]) if self._started else b""
        if pad:
            tail += bytes(keep - len(tail))
        return tail
