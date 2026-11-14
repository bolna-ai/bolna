import numpy as np

from bolna.synthesizer.pcm_shaping import (
    CLAUSE_PAUSE_MS,
    LEAD_MS,
    SENTENCE_PAUSE_MS,
    PcmEdgeTrimmer,
    split_speakable,
)

RATE = 24000


def _speech(ms):
    t = np.arange(RATE * ms // 1000)
    return (8000 * np.sin(2 * np.pi * 220 * t / RATE)).astype(np.int16).tobytes()


def _silence(ms):
    return bytes(RATE * ms // 1000 * 2)


def _ms(audio):
    return len(audio) / 2 / RATE * 1000


def _shape(audio, tail_ms, pad, chunk=3841):
    trimmer = PcmEdgeTrimmer(RATE)
    out = b"".join(trimmer.feed(audio[i : i + chunk]) for i in range(0, len(audio), chunk))
    return out + trimmer.finish(tail_ms, pad=pad)


class TestSplitSpeakable:
    def test_speaks_up_to_the_last_sentence_end(self):
        assert split_speakable("Sure. क्या आप ये") == ("Sure.", "क्या आप ये", SENTENCE_PAUSE_MS)
        assert split_speakable("करना होगा। तोह just") == ("करना होगा।", "तोह just", SENTENCE_PAUSE_MS)

    def test_short_text_with_no_sentence_end_waits_for_its_sentence(self):
        assert split_speakable("So just to confirm, would you") == ("", "So just to confirm, would you", 0)

    def test_a_long_wait_is_cut_at_its_last_clause_end(self):
        tail = "and then " * 12 + "we stop, okay"
        spoken, held, pause = split_speakable(f"Done. {tail}")
        assert (held, pause) == ("okay", CLAUSE_PAUSE_MS) and spoken.endswith("we stop,")

    def test_a_request_never_ends_on_a_code_or_number_before_a_comma(self):
        text = "we have arranged a special offer and " * 3 + "you get it with code BS100, bringing"
        spoken, held, pause = split_speakable(text)
        assert (spoken, held, pause) == (text, "", 0)

    def test_long_text_with_no_boundary_is_spoken_whole(self):
        text = "word " * 30
        assert split_speakable(text) == (text, "", 0)

    def test_a_period_in_an_amount_or_abbreviation_is_not_an_end(self):
        assert split_speakable("It costs Rs. 500 for 2. 5 kg now") == ("", "It costs Rs. 500 for 2. 5 kg now", 0)


class TestPcmEdgeTrimmer:
    def test_the_silence_at_both_edges_is_set(self):
        shaped = _shape(_silence(140) + _speech(1000) + _silence(500), SENTENCE_PAUSE_MS, pad=True)
        assert _ms(shaped) == LEAD_MS + 1000 + SENTENCE_PAUSE_MS

    def test_a_short_tail_is_padded_only_when_asked(self):
        audio = _speech(1000) + _silence(50)
        assert _ms(_shape(audio, 300, pad=True)) == 1300
        assert _ms(_shape(audio, 300, pad=False)) == 1050

    def test_nothing_but_edge_silence_is_removed(self):
        audio = _speech(800) + _silence(820) + _speech(30) + _silence(470) + _speech(50) + _silence(20)
        assert _ms(_shape(audio, 0, pad=False)) == 800 + 820 + 30 + 470 + 50

    def test_the_first_sound_is_passed_on_at_once(self):
        trimmer = PcmEdgeTrimmer(RATE)
        assert _ms(trimmer.feed(_silence(100) + _speech(10))) == LEAD_MS + 10
