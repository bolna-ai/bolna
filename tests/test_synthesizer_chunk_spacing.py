"""Pieces of a streaming LLM reply reach the voice with the space at each cut restored for voices that join
pieces exactly as sent, and unchanged for voices that separate the pieces themselves."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bolna.synthesizer.base_synthesizer import BaseSynthesizer
from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from bolna.synthesizer.deepgram_synthesizer import DeepgramSynthesizer
from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer
from bolna.synthesizer.smallest_synthesizer import SmallestSynthesizer
from bolna.synthesizer.soniox_synthesizer import SonioxSynthesizer

# Cut the way the LLM wrappers cut: at the last space in the buffer, which is dropped.
PIECES = ["Your appointment is confirmed for", "Monday morning at the main branch."]
# A token longer than the buffer has no space to cut at, so it arrives in parts.
URL_PIECES = ["Track your order at", "https://example.com/orders/2026/", "abc123 anytime."]

# Join pieces exactly as sent.
GLUING_VOICES = {
    "cartesia": (CartesiaSynthesizer, {"voice_id": "v", "voice": "v"}),
    "deepgram": (DeepgramSynthesizer, {"voice_id": "v", "voice": "v"}),
    "soniox": (SonioxSynthesizer, {"voice": "v"}),
}
# Separate the pieces they receive themselves.
SPACING_VOICES = {
    "sarvam": (SarvamSynthesizer, {"voice_id": "v", "model": "bulbul:v3", "language": "hi-IN"}),
    "smallest": (SmallestSynthesizer, {"voice_id": "v"}),
}


def _voice(cls, kwargs):
    tm = MagicMock()
    tm.is_sequence_id_in_current_ids.return_value = True
    synth = cls(**kwargs, synthesizer_key="key", stream=True, task_manager_instance=tm)
    synth.sender = AsyncMock()
    return synth


async def _push_reply(synth, pieces):
    for i, piece in enumerate(pieces):
        meta = {"sequence_id": 1, "turn_id": 1, "end_of_llm_stream": i == len(pieces) - 1}
        await synth.push({"data": piece, "meta_info": meta})
        await synth.sender_task
    return [call.args[0] for call in synth.sender.await_args_list]


@pytest.mark.parametrize("name", GLUING_VOICES)
async def test_gluing_voices_get_the_space_back_at_each_cut(name):
    synth = _voice(*GLUING_VOICES[name])
    sent = await _push_reply(synth, PIECES)
    assert "".join(sent).strip() == " ".join(PIECES)
    assert [m["text"] for m in synth.text_queue] == PIECES  # transcripts and marks keep the LLM text


@pytest.mark.parametrize("name", GLUING_VOICES)
async def test_a_long_token_cut_midway_gets_no_space(name):
    sent = await _push_reply(_voice(*GLUING_VOICES[name]), URL_PIECES)
    assert "".join(sent).strip() == "Track your order at https://example.com/orders/2026/abc123 anytime."


@pytest.mark.parametrize("name", SPACING_VOICES)
async def test_voices_that_space_pieces_themselves_get_them_as_cut(name):
    assert await _push_reply(_voice(*SPACING_VOICES[name]), PIECES) == PIECES


def test_text_chunker_keeps_numbers_abbreviations_and_links_whole():
    text = "The total is ₹2,50,000 at 3.5% from 10:30 am, i.e. via example.com for 2-3 weeks. Thanks!"
    chunks = list(BaseSynthesizer.text_chunker(None, text))
    whole = {"₹2,50,000 ", "3.5% ", "10:30 ", "am, ", "i.e. ", "example.com ", "2-3 ", "weeks. ", "Thanks! "}
    assert "".join(chunks) == text + " "
    assert whole <= set(chunks)
