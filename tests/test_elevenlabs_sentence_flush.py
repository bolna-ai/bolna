"""The ElevenLabs multi-stream sender flushes at sentence ends, so ElevenLabs never picks a mid-sentence cut itself."""

import json
from types import SimpleNamespace

import pytest
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
            "आज मौसम काफी अच्छा है, तो हम शाम को पार्क में चलेंगे। फिर वहाँ से हम",
            "बाजार जाएंगे और कुछ फल खरीदेंगे। क्या आप भी हमारे साथ",
            "चलना चाहेंगे?",
        ],
    )

    assert _flushed_segments(synth.websocket.sent) == [
        "आज मौसम काफी अच्छा है, तो हम शाम को पार्क में चलेंगे।",
        "फिर वहाँ से हम बाजार जाएंगे और कुछ फल खरीदेंगे।",
        "क्या आप भी हमारे साथ चलना चाहेंगे?",
        "",
    ]
    assert synth.websocket.sent[-1] == {"context_id": "ctx-live", "close_context": True}
    assert synth.pending_text == ""


async def test_each_sentence_in_one_push_gets_its_own_flush():
    synth = _make_synth()
    await _turn(synth, ["ठीक है, मैंने details आपके email पर भेज दी हैं। आप इन्हें कभी भी देख सकते हैं। क्या मैं कुछ और बताऊँ?"])

    assert _flushed_segments(synth.websocket.sent) == [
        "ठीक है, मैंने details आपके email पर भेज दी हैं।",
        "आप इन्हें कभी भी देख सकते हैं।",
        "क्या मैं कुछ और बताऊँ?",
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


@pytest.mark.parametrize(
    "sentence",
    [
        "आपको Rs. 500 का discount मिलेगा।",
        "Dr. Sharma कल आपको call करेंगे।",
        "Option No. 2 सबसे अच्छा है।",
        "1. पहला plan और 2. दूसरा plan देखिए।",
        "Mr. Rao will call you tomorrow.",
        "The total covers taxes, fees etc. and delivery charges.",
    ],
)
async def test_a_dot_inside_a_sentence_does_not_end_it(sentence):
    synth = _make_synth()
    await _turn(synth, [sentence + " क्या मैं आगे बताऊँ?"])

    assert _flushed_segments(synth.websocket.sent) == [sentence, "क्या मैं आगे बताऊँ?", ""]


async def test_a_dot_before_a_hindi_word_ends_the_sentence():
    synth = _make_synth()
    await _turn(synth, ["कल बारिश हो सकती है. छाता साथ रखिए।"])

    assert _flushed_segments(synth.websocket.sent) == ["कल बारिश हो सकती है.", "छाता साथ रखिए।", ""]


@pytest.mark.parametrize(
    "reply, segments",
    [
        ("您好。我们的产品很受欢迎，您要了解吗？", ["您好。", "我们的产品很受欢迎，您要了解吗？", ""]),
        ("مرحبا، هل تريد معرفة المزيد؟ نعم", ["مرحبا، هل تريد معرفة المزيد؟", "نعم"]),
        ("شکریہ۔ آپ کیسے ہیں؟", ["شکریہ۔", "آپ کیسے ہیں؟", ""]),
    ],
)
async def test_other_scripts_are_flushed_at_their_own_sentence_marks(reply, segments):
    synth = _make_synth()
    await _turn(synth, [reply])

    assert _flushed_segments(synth.websocket.sent) == segments


async def test_long_text_without_a_sentence_end_flushes_at_the_last_comma():
    synth = _make_synth()
    clause = "this clause keeps going without ever ending the sentence, "
    text = clause * (MAX_UNFLUSHED_CHARS // len(clause) + 1)
    await synth.sender(text + "and then some", sequence_id=1)

    assert _flushed_segments(synth.websocket.sent) == [text.strip()]
    assert synth.pending_text == "and then some"


@pytest.mark.parametrize(
    "text, cut",
    [
        ("很" * (MAX_UNFLUSHED_CHARS + 20), MAX_UNFLUSHED_CHARS),
        ("很" * 200 + "，" + "好" * 100, 201),
    ],
)
async def test_long_text_without_spaces_is_cut_at_a_comma_or_the_cap(text, cut):
    synth = _make_synth()
    await synth.sender(text, sequence_id=1)

    assert _flushed_segments(synth.websocket.sent) == [text[:cut]]
    assert synth.pending_text == text[cut:]


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
