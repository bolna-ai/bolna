"""What push() hands each voice's sender(), with expressive tags off (unchanged) and on (tags handled)."""

from unittest.mock import AsyncMock, MagicMock

import bolna.agent_manager.task_manager as tmmod

import pytest

from bolna.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from bolna.synthesizer.deepgram_synthesizer import DeepgramSynthesizer
from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer, ElevenlabsV3Synthesizer
from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer
from bolna.synthesizer.smallest_synthesizer import SmallestSynthesizer
from bolna.synthesizer.soniox_synthesizer import SonioxSynthesizer
from bolna.synthesizer.synthesizer_pool import SynthesizerPool

VOICES = {
    "cartesia": lambda **kw: CartesiaSynthesizer(voice_id="v", voice="v", language="en", model="sonic-3.6", **kw),
    "elevenlabs_v2_5": lambda **kw: ElevenlabsSynthesizer(voice="v", voice_id="v", model="eleven_turbo_v2_5", **kw),
    "elevenlabs_v4": lambda **kw: ElevenlabsV3Synthesizer(voice="v", voice_id="v", model="eleven_v4_turbo", **kw),
    "sarvam": lambda **kw: SarvamSynthesizer(model="bulbul:v3", voice_id="v", language="hi-IN", **kw),
    "deepgram": lambda **kw: DeepgramSynthesizer(voice_id="zeus", voice="zeus", **kw),
    "smallest": lambda **kw: SmallestSynthesizer(voice_id="candela", **kw),
    "soniox": lambda **kw: SonioxSynthesizer(voice="Adrian", **kw),
}

# How the LLM wrappers cut a reply: at the last space, which they drop.
PIECES = [
    "<<apologetic>> Sorry for the wait. [[pause",
    '400]] Your ID BK4821 is ready <break time="1s"/>',
    "[calm] ok.",
]


def _voice(name, **kw):
    tm = MagicMock()
    tm.is_sequence_id_in_current_ids.return_value = True
    synth = VOICES[name](synthesizer_key="k", stream=True, task_manager_instance=tm, **kw)
    synth.sender = AsyncMock()
    return synth


async def _sent(synth, pieces, sequence_id=1):
    for i, piece in enumerate(pieces):
        meta = {"sequence_id": sequence_id, "turn_id": sequence_id, "end_of_llm_stream": i == len(pieces) - 1}
        await synth.push({"data": piece, "meta_info": meta})
        if synth.sender_task:
            await synth.sender_task
    return [call.args[0] for call in synth.sender.await_args_list]


@pytest.mark.parametrize("name", VOICES)
async def test_with_the_setting_off_every_voice_gets_each_piece_unchanged(name):
    assert await _sent(_voice(name), PIECES) == PIECES


async def test_with_the_setting_on_a_voice_without_tag_support_gets_words_only():
    assert await _sent(_voice("sarvam", expressive_tags=True), PIECES) == [
        " Sorry for the wait. ",  # "[[pause" waits for the next piece
        " Your ID BK4821 is ready ",  # the joined [[pause 400]] and the raw <break …/> are removed
        " ok.",  # [calm] is a known tag word, so it goes entirely
    ]


async def test_a_piece_that_is_all_half_tag_is_sent_with_the_next_piece():
    pieces = ["Your code is", "<<spell:", "BK4821>> done."]
    assert await _sent(_voice("sarvam", expressive_tags=True), pieces) == ["Your code is", "BK4821 done."]


async def test_a_new_turn_drops_a_half_tag_held_from_the_last_one():
    synth = _voice("sarvam", expressive_tags=True)
    await synth.push({"data": "Hello [[pause", "meta_info": {"sequence_id": 1, "end_of_llm_stream": False}})
    await synth.sender_task
    assert await _sent(synth, ["Hi there."], sequence_id=2) == ["Hello ", "Hi there."]


async def test_with_the_setting_on_a_provider_tag_the_voice_performs_is_kept():
    pieces = ['Hi <break time="1s"/> there [[pause 400]] ok']
    assert await _sent(_voice("cartesia", expressive_tags=True), pieces) == [
        'Hi <break time="1s"/> there <break time="400ms"/> ok'
    ]


# The same reply, cut by the LLM inside a tag, as each voice receives it.
REPLY = ["<<apologetic>> Sorry for the wait. [[pause", "400]] Your ID BK4821 is ready. ^^Thanks^^ [[laugh]]"]


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "cartesia",
            [
                " Sorry for the wait. ",
                ' <break time="400ms"/> Your ID <spell>BK4821</spell> is ready. Thanks [laughter] ',
            ],
        ),
        ("elevenlabs_v2_5", [" Sorry for the wait. ", ' <break time="0.4s" /> Your ID B K 4 8 2 1 is ready. Thanks ']),
        (
            "elevenlabs_v4",
            # The [[laugh]] that ends the reply is dropped: v4 can go silent on a reply that ends on a sound.
            ["[apologetic] Sorry for the wait. ", " [short pause] Your ID B K 4 8 2 1 is ready. THANKS "],
        ),
        ("sarvam", [" Sorry for the wait. ", " Your ID BK4821 is ready. Thanks "]),
    ],
)
async def test_each_voice_gets_the_reply_in_its_own_syntax(name, expected):
    assert await _sent(_voice(name, expressive_tags=True), REPLY) == expected


async def test_cartesia_speed_spans_return_to_the_agents_own_speed():
    synth = _voice("cartesia", expressive_tags=True, speed=1.2)
    sent = await _sent(synth, ["Note <<slow: 4 7 2 9>> please."])
    assert sent == ['Note <speed ratio="0.96"/> 4 7 2 9 <speed ratio="1.2"/> please.']


def _wrapper_pieces(text, buffer_size, token=4):
    """The LLM wrappers' cut: at buffer_size, split at the last space and drop it."""
    pieces, buffer = [], ""
    for i in range(0, len(text), token):
        buffer += text[i : i + token]
        if len(buffer) >= buffer_size:
            split = buffer.rsplit(" ", 1)
            pieces.append(split[0])
            buffer = split[1] if len(split) > 1 else ""
    return pieces + [buffer]


@pytest.mark.parametrize("buffer_size", range(20, 61))
async def test_a_tag_the_llm_cut_comes_back_whole_at_every_allowed_buffer_size(buffer_size):
    text = 'Hi there <break time="1s"/> ok ^^ज़रूर^^ <speed ratio="0.8"/> fine'
    synth = _voice("cartesia", expressive_tags=True, buffer_size=buffer_size)
    sent = " ".join(await _sent(synth, _wrapper_pieces(text, buffer_size)))
    assert '<break time="1s"/>' in sent and '<speed ratio="0.8"/>' in sent and "ज़रूर" in sent


async def test_every_canned_message_gets_its_own_pause_budget():
    synth = _voice("cartesia", expressive_tags=True)
    message = "One [[pause]] two [[pause]] three [[pause]] four."
    first = await _sent(synth, [message], sequence_id=-1)
    synth.reset_turn_state()  # the first message finished playing
    second = await _sent(synth, [message], sequence_id=-1)
    assert first[0].count("<break") == 3 and second[-1].count("<break") == 3


async def test_billing_and_end_of_turn_matching_count_only_the_spoken_words():
    synth = _voice("elevenlabs_v4", expressive_tags=True)
    await _sent(synth, ["<<warm>> Hi [[pause 400]] there [smiling]"])
    assert synth.synthesized_characters == len(" Hi there ") and synth.current_text == " Hi there "


def test_elevenlabs_v2_5_parses_ssml_only_with_the_setting_on():
    tm = MagicMock()
    off = ElevenlabsSynthesizer(
        voice="v", voice_id="v", model="eleven_turbo_v2_5", synthesizer_key="k", task_manager_instance=tm
    )
    on = ElevenlabsSynthesizer(
        voice="v",
        voice_id="v",
        model="eleven_turbo_v2_5",
        synthesizer_key="k",
        task_manager_instance=tm,
        expressive_tags=True,
    )
    assert "enable_ssml_parsing" not in off.ws_url
    assert on.ws_url.endswith("&enable_ssml_parsing=true")


def test_every_language_voice_gets_the_agents_setting(monkeypatch):
    voices = {"cartesia": MagicMock(), "sarvam": MagicMock()}
    monkeypatch.setattr(tmmod, "SynthesizerPool", MagicMock())
    monkeypatch.setattr(tmmod, "SUPPORTED_SYNTHESIZER_MODELS", voices)
    tm = MagicMock()
    tm.turn_based_conversation, tm.is_web_based_call, tm.enforce_streaming = False, False, True
    tm.kwargs, tm.provider_api_keys, tm.language_switcher = {}, {}, None
    tm._TaskManager__pool_leg_kwargs = tmmod.TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, tmmod.TaskManager)
    tm._TaskManager__language_switch_enabled = MagicMock(return_value=False)
    tm.task_config = {
        "tools_config": {
            "transcriber": {"provider": "deepgram", "language": "en"},
            "synthesizer": {
                "provider": "cartesia",
                "expressive_tags": True,
                "multilingual": {
                    "en": {"provider": "cartesia", "provider_config": {"voice": "v", "voice_id": "v"}},
                    "hi": {"provider": "sarvam", "provider_config": {"voice": "v", "voice_id": "v"}},
                },
            },
            "output": {"provider": "plivo"},
            "llm_agent": None,
        }
    }
    tmmod.TaskManager._TaskManager__setup_synthesizer.__get__(tm, tmmod.TaskManager)()
    assert voices["cartesia"].call_args.kwargs["expressive_tags"] is True
    assert voices["sarvam"].call_args.kwargs["expressive_tags"] is True


async def test_after_a_language_switch_the_new_voice_never_speaks_a_tag():
    english, hindi = _voice("cartesia", expressive_tags=True), _voice("sarvam", expressive_tags=True)
    pool = SynthesizerPool(synthesizers={"en": english, "hi": hindi}, active_label="en", multilingual_config={})
    reply = '<<warm>> Hi [warmly] there <break time="1s"/> [[pause 300]] friend.'
    await pool.push({"data": reply, "meta_info": {"sequence_id": 1, "end_of_llm_stream": True}})
    await english.sender_task
    pool.active_label = "hi"  # the language switch
    await pool.push({"data": reply, "meta_info": {"sequence_id": 2, "end_of_llm_stream": True}})
    await hindi.sender_task
    assert english.sender.await_args.args[0] == ' Hi there <break time="1s"/> friend.'
    assert hindi.sender.await_args.args[0] == " Hi there friend."


async def test_two_pauses_stay_one_even_when_the_llm_cut_falls_between_them():
    synth = _voice("cartesia", expressive_tags=True)
    sent = await _sent(synth, ['Done. <break time="1s"/>', "[[pause 400]] Next part."])
    assert " ".join(sent).count("<break") == 1


async def test_a_cartesia_agent_without_a_speed_setting_still_gets_its_tags():
    synth = _voice("cartesia", expressive_tags=True, speed=None)
    assert await _sent(synth, ["Note <<slow: 4 7 2 9>> please."]) == [
        'Note <speed ratio="0.8"/> 4 7 2 9 <speed ratio="1.0"/> please.'
    ]


async def test_a_canned_line_never_carries_a_tag_over_to_the_next_one():
    synth = _voice("cartesia", expressive_tags=True)
    for text, category, end in (
        ("Connecting you now. <<slow: 98765", "handoff", True),
        ("Thank you, goodbye.", None, True),
    ):
        meta = {"sequence_id": -1, "turn_id": None, "end_of_llm_stream": end, "message_category": category}
        await synth.push({"data": text, "meta_info": meta})
        if synth.sender_task:
            await synth.sender_task
    assert synth.sender.await_args_list[-1].args[0] == "Thank you, goodbye."


async def test_elevenlabs_v4_never_gets_a_reply_that_ends_on_a_sound():
    pieces = ["<<warm>> Your booking is confirmed. [[pause 400]] See you", "on Monday. [[laugh]]"]
    sent = await _sent(_voice("elevenlabs_v4", expressive_tags=True), pieces)
    assert "[laughs]" not in sent[-1] and sent[-1].strip().endswith("Monday.")
