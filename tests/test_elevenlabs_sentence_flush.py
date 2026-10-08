"""The ElevenLabs multi-stream sender flushes at sentence ends, so ElevenLabs never picks a
mid-sentence cut itself (where it can voice the word at the cut twice, e.g. "और और")."""

import json
from types import SimpleNamespace

from websockets.protocol import State

from bolna.synthesizer.elevenlabs_synthesizer import MAX_UNFLUSHED_CHARS, ElevenlabsSynthesizer


class FakeWS:
    def __init__(self):
        self.sent = []
        self.state = State.OPEN

    async def send(self, data):
        self.sent.append(json.loads(data))


def _make_synth():
    synth = ElevenlabsSynthesizer(
        voice="v",
        voice_id="vid",
        synthesizer_key="test",
        caching=False,
        task_manager_instance=SimpleNamespace(is_sequence_id_in_current_ids=lambda sid: True),
    )
    synth.context_id = "ctx-live"
    synth.websocket = FakeWS()
    return synth


def _flushed_segments(sent):
    """Text sent between flushes, one entry per flush."""
    segments, words = [], []
    for message in sent:
        if message.get("flush"):
            segments.append("".join(words).strip())
            words = []
        elif message.get("text"):
            words.append(message["text"])
    return segments


async def _turn(synth, pushes):
    for text in pushes:
        await synth.sender(text, sequence_id=1)
    await synth.sender("", sequence_id=1, end_of_llm_stream=True)


async def test_hindi_turn_is_flushed_at_sentence_ends_not_mid_sentence():
    """Pushed in pieces the way the LLM streams it, with the boundary space dropped."""
    synth = _make_synth()
    await _turn(
        synth,
        [
            "हम twenty years से भी ज्यादा समय से आपकी सेवा कर रहे हैं, तो आप बिल्कुल सही हाथों में हैं। और Smart",
            "Saver Plan हमारे सबसे popular plans में से एक है, पिछले महीने four thousand",
            "लोगों ने इसे चुना है। क्या मैं आपको इस plan के बारे में कुछ information",
            "बताऊँ?",
        ],
    )

    assert _flushed_segments(synth.websocket.sent) == [
        "हम twenty years से भी ज्यादा समय से आपकी सेवा कर रहे हैं, तो आप बिल्कुल सही हाथों में हैं।",
        "और Smart Saver Plan हमारे सबसे popular plans में से एक है, पिछले महीने four thousand लोगों ने इसे चुना है।",
        "क्या मैं आपको इस plan के बारे में कुछ information बताऊँ?",
        "",
    ]
    assert synth.websocket.sent[-1] == {"context_id": "ctx-live", "close_context": True}
    assert synth.pending_text == ""


async def test_each_sentence_in_one_push_gets_its_own_flush():
    """Sent as one batch, ElevenLabs cut this reply right after "journey में और"."""
    synth = _make_synth()
    reply = (
        "Perfect, मैंने checkout link आपके WhatsApp पर फिर से भेज दिया है। वैसे, हम आपकी fitness journey में और "
        "कैसे help कर सकते हैं। आपका कोई और question हो जिसमें मैं help कर सकूँ?"
    )
    await _turn(synth, [reply])

    assert _flushed_segments(synth.websocket.sent) == [
        "Perfect, मैंने checkout link आपके WhatsApp पर फिर से भेज दिया है।",
        "वैसे, हम आपकी fitness journey में और कैसे help कर सकते हैं।",
        "आपका कोई और question हो जिसमें मैं help कर सकूँ?",
        "",
    ]


async def test_english_turn_is_flushed_at_full_stops():
    synth = _make_synth()
    await _turn(synth, ["Thanks for calling. I have", "pulled up your account. Anything", "else?"])

    assert _flushed_segments(synth.websocket.sent) == [
        "Thanks for calling.",
        "I have pulled up your account.",
        "Anything else?",
        "",
    ]


async def test_text_without_a_sentence_end_waits_for_the_next_push():
    synth = _make_synth()
    await synth.sender("Thanks for calling, I have pulled up your", sequence_id=1)

    assert synth.websocket.sent == []
    assert synth.pending_text == "Thanks for calling, I have pulled up your"


async def test_decimal_point_is_not_a_sentence_end():
    synth = _make_synth()
    await synth.sender("You get 7.5 percent off", sequence_id=1)

    assert synth.websocket.sent == []
    assert synth.pending_text == "You get 7.5 percent off"


async def test_long_text_without_a_sentence_end_flushes_at_the_last_comma():
    synth = _make_synth()
    clause = "this clause keeps going without ever ending the sentence, "
    text = clause * (MAX_UNFLUSHED_CHARS // len(clause) + 1)
    await synth.sender(text + "and then some", sequence_id=1)

    assert _flushed_segments(synth.websocket.sent) == [text.strip()]
    assert synth.pending_text == "and then some"


async def test_interruption_drops_the_unsent_half_sentence():
    synth = _make_synth()
    await synth.sender("Sure, I can help. Let me check the", sequence_id=1)
    await synth.handle_interruption()

    synth.context_id = "ctx-next"
    synth.websocket.sent.clear()
    await synth.sender("What would you like to know?", sequence_id=2, end_of_llm_stream=True)

    assert _flushed_segments(synth.websocket.sent) == ["What would you like to know?"]


async def test_stale_sequence_drops_the_unsent_half_sentence():
    synth = _make_synth()
    await synth.sender("Sure, I can help. Let me check the", sequence_id=1)
    synth.task_manager_instance = SimpleNamespace(is_sequence_id_in_current_ids=lambda sid: False)

    await synth.sender("details for you.", sequence_id=1)

    assert synth.pending_text == ""
