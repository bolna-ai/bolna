"""Smallest voices every request on its own, so text cut mid-sentence is heard as a break in the reply."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
import websockets

from bolna.synthesizer import smallest_synthesizer
from bolna.synthesizer.smallest_synthesizer import SmallestSynthesizer


class RecordingSocket:
    state = websockets.protocol.State.OPEN

    def __init__(self):
        self.texts = []

    async def send(self, raw):
        await asyncio.sleep(0)  # a real send can yield, letting later pushes run
        self.texts.append(json.loads(raw)["text"])


def make_synth():
    synth = SmallestSynthesizer(
        voice_id="candela", stream=True, synthesizer_key="test", task_manager_instance=MagicMock()
    )
    synth.websocket = RecordingSocket()
    return synth


async def push_reply(synth, pieces, sequence_id=1):
    """Each LLM buffer flush becomes its own sender task, as in _push_stream."""
    last = len(pieces) - 1
    tasks = [
        asyncio.create_task(synth.sender(text, sequence_id, end_of_llm_stream=i == last))
        for i, text in enumerate(pieces)
    ]
    await asyncio.gather(*tasks)
    return synth.websocket.texts


async def test_reply_cut_mid_sentence_is_sent_as_whole_sentences():
    synth = make_synth()
    sent = await push_reply(synth, ["I can help with admissions and fees. Please let me know what you", "need."])
    assert sent == ["I can help with admissions and fees.", "Please let me know what you need."]


@pytest.mark.parametrize(
    "text",
    [
        "Let me check that for you",
        "Your appointment is with Dr. Rao at",
        "The annual tuition fee is Rs. 50,000 for the general category and the hostel",
    ],
)
async def test_text_without_a_sentence_end_is_sent_at_once(text):
    synth = make_synth()
    await synth.sender(text, 1, end_of_llm_stream=False)
    assert synth.websocket.texts == [text]


async def test_held_text_is_sent_when_no_more_text_follows(monkeypatch):
    monkeypatch.setattr(smallest_synthesizer, "HELD_TEXT_MAX_WAIT_S", 0.01)
    synth = make_synth()
    await synth.sender("Sure. Let me check", 1, end_of_llm_stream=False)
    assert synth.websocket.texts == ["Sure."]
    await asyncio.sleep(0.05)
    assert synth.websocket.texts == ["Sure.", "Let me check"]


async def test_no_request_exceeds_smallest_limit():
    synth = make_synth()
    sentence = "This sentence is about sixty characters long, give or take. "
    sent = await push_reply(synth, [sentence * 7])
    assert len(sent) > 1
    assert all(len(text) <= 250 and text.endswith(".") for text in sent)
    assert " ".join(sent) == (sentence * 7).strip()


async def test_interrupted_reply_never_speaks_its_held_text(monkeypatch):
    monkeypatch.setattr(smallest_synthesizer, "HELD_TEXT_MAX_WAIT_S", 0.01)
    synth = make_synth()
    current = {1}
    synth.task_manager_instance.is_sequence_id_in_current_ids.side_effect = lambda seq: seq in current
    await synth.sender("Done. And then", 1, end_of_llm_stream=False)
    current.clear()
    current.add(2)  # caller barged in; the next reply has a new sequence
    await synth.sender("Hi.", 2, end_of_llm_stream=True)
    await asyncio.sleep(0.05)
    assert synth.websocket.texts == ["Done.", "Hi."]
