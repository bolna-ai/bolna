"""ElevenLabs gets each reply one sentence at a time, so every generation boundary is a sentence boundary."""

import json

import pytest
import websockets

from bolna.synthesizer.elevenlabs_synthesizer import (
    CLAUSE_FLUSH_CHARS,
    MAX_UNFLUSHED_CHARS,
    ElevenlabsSynthesizer,
    split_flushable,
)


@pytest.mark.parametrize(
    "text, units, rest",
    [
        (
            "आप बिल्कुल सही हाथों में हैं। और Pro Performance",
            ["आप बिल्कुल सही हाथों में हैं।"],
            "और Pro Performance",
        ),
        ("Your order is on its way. Anything else?", ["Your order is on its way.", "Anything else?"], ""),
        (
            "我们的产品在全国都非常受欢迎，很多老客户都会反复购买。您要了解一下吗？",
            ["我们的产品在全国都非常受欢迎，很多老客户都会反复购买。", "您要了解一下吗？"],
            "",
        ),
        ("Ji haan, maine link bhej diya hai. toh aap", ["Ji haan, maine link bhej diya hai."], "toh aap"),
        (
            "Here are the options for you: 1. Lodha Group with modern homes. 2. Hiranandani",
            ["Here are the options for you: 1. Lodha Group with modern homes."],
            "2. Hiranandani",
        ),
        ("Your order number is 4821. It will reach you", ["Your order number is 4821."], "It will reach you"),
        ("Please write to support@example.com. We reply", ["Please write to support@example.com."], "We reply"),
    ],
)
def test_text_is_split_at_sentence_ends(text, units, rest):
    assert split_flushable(text, end_of_stream=False) == (units, rest)


def test_a_sentence_ending_exactly_at_the_push_boundary_is_sent_without_waiting():
    assert split_flushable("Your order has been shipped today.", end_of_stream=False) == (
        ["Your order has been shipped today."],
        "",
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "Hello Mr. Sharma, your order has shipped.",
        "Namaste Dr. Gupta ji, aapka appointment kal hai.",
        "You get 7.5 percent off on this order today.",
        "Orders above Rs. 500 ship free in the U.S. this week.",
        "Option No. 2 is the cheapest plan we have.",
        "I am calling from Acme Industries Pvt. Ltd. about your order.",
        "Please ask our manager (Mr. Sharma) for the refund today.",
        "Your appointment is on Jan. 15 at 10 a.m. at our office.",
        "Visit https://example.com/?ref=abc for more on the offer today.",
    ],
)
def test_dots_inside_a_sentence_do_not_split_it(sentence):
    assert split_flushable(sentence, end_of_stream=False) == ([sentence], "")


def test_short_sentences_are_sent_on_their_own():
    assert split_flushable("जी। बिल्कुल। मैं अभी आपका order check करती हूँ। हाँ,", end_of_stream=False) == (
        ["जी।", "बिल्कुल।", "मैं अभी आपका order check करती हूँ।"],
        "हाँ,",
    )


def test_end_of_stream_sends_everything_held():
    assert split_flushable("जी। और कुछ", end_of_stream=True) == (["जी।", "और कुछ"], "")


def test_long_unfinished_sentence_is_cut_at_its_last_clause_break_outside_numbers():
    head = "Aapka total amount Rs 1,25,000 ho gaya hai, jo aapke account se debit hoga"
    tail = " aur iske saath aapko ek free gift bhi milega jo hum aapke address par bhej denge"
    units, rest = split_flushable(head + tail, end_of_stream=False)
    assert len(head + tail) > CLAUSE_FLUSH_CHARS
    assert units == ["Aapka total amount Rs 1,25,000 ho gaya hai,"]
    assert rest.startswith("jo aapke account")


def test_unfinished_sentence_without_a_clause_break_is_held_until_the_cap():
    text = "word " * (CLAUSE_FLUSH_CHARS // 5 + 2)
    assert split_flushable(text, end_of_stream=False) == ([], text.strip())


@pytest.mark.parametrize("text", ["word " * 160, "很" * 700])
def test_neither_sent_units_nor_held_text_exceed_the_cap(text):
    units, rest = split_flushable(text, end_of_stream=False)
    assert units
    assert all(len(unit) <= MAX_UNFLUSHED_CHARS for unit in units)
    assert len(rest) <= MAX_UNFLUSHED_CHARS


class FakeSocket:
    state = websockets.protocol.State.OPEN

    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))


class CurrentSequences:
    def __init__(self, current):
        self.current = set(current)

    def is_sequence_id_in_current_ids(self, sequence_id):
        return sequence_id in self.current


def _synth(current=(1,)):
    synth = ElevenlabsSynthesizer(
        voice="v",
        voice_id="vid",
        synthesizer_key="test",
        caching=False,
        task_manager_instance=CurrentSequences(current),
    )
    synth.websocket = FakeSocket()
    synth.context_id = "ctx"
    return synth


def _spoken(synth):
    """Sentences sent, each checked to carry its own flush."""
    texts = [m for m in synth.websocket.sent if m.get("text", "").strip()]
    assert all(m.get("flush") for m in texts)
    return [m["text"].strip() for m in texts]


async def test_each_sentence_goes_out_as_one_flushed_frame():
    synth = _synth()
    await synth.sender("Perfect, link अभी WhatsApp पर भेज देते हैं. आपका offer सिर्फ अगले कुछ", 1)
    await synth.sender("hours के लिए है। क्या मैं कुछ और", 1)
    await synth.sender("बताऊँ?", 1, end_of_llm_stream=True)

    assert _spoken(synth) == [
        "Perfect, link अभी WhatsApp पर भेज देते हैं.",
        "आपका offer सिर्फ अगले कुछ hours के लिए है।",
        "क्या मैं कुछ और बताऊँ?",
    ]
    assert synth.websocket.sent[-1]["close_context"]


async def test_a_superseded_turns_unsent_text_is_dropped():
    synth = _synth(current=(1, 2))
    await synth.sender("Sure, let me check the order status for", 1)
    synth.task_manager_instance.current = {2}
    await synth.sender("Your refund was processed yesterday.", 2, end_of_llm_stream=True)

    assert _spoken(synth) == ["Your refund was processed yesterday."]


async def test_a_live_turns_unsent_text_is_spoken_before_the_follow_up():
    synth = _synth(current=(1, 2))
    await synth.sender("Sure, let me check the order status for you", 1)
    await synth.sender("Your refund was processed yesterday.", 2, end_of_llm_stream=True)

    assert _spoken(synth) == ["Sure, let me check the order status for you", "Your refund was processed yesterday."]


async def test_first_audio_latency_is_measured_from_the_first_push_even_when_it_is_held():
    synth = _synth()
    await synth.sender("Sure, let me check the order status for", 1)

    assert synth.websocket.sent == []
    assert synth.ws_send_time is not None


async def test_interruption_drops_the_unsent_text():
    synth = _synth()
    await synth.sender("Sure, let me check the order status for", 1)
    await synth.handle_interruption()
    synth.websocket.sent.clear()
    synth.context_id = "ctx-next"
    await synth.sender("Okay, what would you like to know?", 1, end_of_llm_stream=True)

    assert _spoken(synth) == ["Okay, what would you like to know?"]
