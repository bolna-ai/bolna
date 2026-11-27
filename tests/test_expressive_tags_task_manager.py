"""The call around the voice when expressive tags are on: transcript, interruptions, stored audio and the prompt."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers import tts_clips
from bolna.helpers.utils import tts_voice_identity
from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer, ElevenlabsV3Synthesizer
from bolna.synthesizer.sarvam_synthesizer import SarvamSynthesizer
from bolna.synthesizer.synthesizer_pool import SynthesizerPool

REPLY = "<<apologetic>> Sorry for the wait. [[pause 400]] Your booking BK4821 is confirmed [smiling]."


def _tm(expressive_tags, synth=None):
    tm = TaskManager.__new__(TaskManager)
    tm.expressive_tags = expressive_tags
    tm.task_config = {"task_type": "conversation"}
    tm.tools = {"synthesizer": synth}
    return tm


def test_the_saved_transcript_has_no_tags_while_the_llms_history_keeps_them():
    history = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "Is my booking done? [Gold]"},
        {"role": "assistant", "content": REPLY},
    ]
    on = _tm(True)._TaskManager__output_messages(history)
    assert on[-1]["content"] == "Sorry for the wait. Your booking BK4821 is confirmed."
    assert on[1]["content"] == "Is my booking done? [Gold]"  # the caller's words are never touched
    assert history[-1]["content"] == REPLY
    assert _tm(False)._TaskManager__output_messages(history)[-1]["content"] == REPLY


def test_an_interrupted_reply_keeps_what_was_heard_when_it_carried_tags():
    heard = "Sorry for the wait. Your booking"
    assert _tm(True).update_transcript_for_interruption(REPLY, heard) == "Sorry for the wait. Your booking"


def test_stored_audio_is_keyed_apart_once_tags_are_on():
    config = {"voice": "Nila", "voice_id": "v1", "model": "eleven_flash_v2_5"}
    assert tts_voice_identity("elevenlabs", config) == tts_voice_identity("elevenlabs", config, expressive_tags=False)
    assert tts_voice_identity("elevenlabs", config) != tts_voice_identity("elevenlabs", config, expressive_tags=True)


@pytest.fixture
def storage(monkeypatch):
    load = MagicMock()
    monkeypatch.setattr("bolna.agent_manager.task_manager.load_fixed_line_clip", load)
    tts_clips._clips.clear()
    return load


async def test_a_fixed_line_with_tags_is_spoken_live_and_never_stored(storage):
    synth = MagicMock(one_shot_matches_stream=True)
    tm = _tm(True, synth)
    tm.assistant_id, tm.fixed_line_categories, tm.fixed_line_voices = "agent", {"agent_hangup"}, {None: "voice"}
    tm.tools["output"] = SimpleNamespace(get_provider=lambda: "plivo")
    tm.turn_based_conversation, tm.is_web_based_call = False, False
    meta = {"message_category": "agent_hangup", "sequence_id": -1}
    assert await tm._TaskManager__play_fixed_line_clip("<<warm>> Thank you, goodbye.", meta) is False
    storage.assert_not_called()


def _voice(cls, model):
    tm = MagicMock()
    return cls(
        voice="v", voice_id="v", model=model, synthesizer_key="k", task_manager_instance=tm, expressive_tags=True
    )


def test_the_llm_is_taught_only_the_tags_its_voice_performs():
    v4 = _tm(True, _voice(ElevenlabsV3Synthesizer, "eleven_v4_turbo"))._TaskManager__with_expressive_tags_note("")
    assert v4.startswith("### Note:\n") and "<<warm>>" in v4 and "[[pause" in v4
    turbo = _tm(True, _voice(ElevenlabsSynthesizer, "eleven_turbo_v2_5"))._TaskManager__with_expressive_tags_note("x\n")
    assert turbo.startswith("x\n") and "[[pause" in turbo and "<<warm>>" not in turbo
    assert (
        _tm(False, _voice(ElevenlabsSynthesizer, "eleven_v4_turbo"))._TaskManager__with_expressive_tags_note("") == ""
    )


def test_a_language_pool_teaches_the_tags_of_the_voice_speaking_now():
    hindi = SarvamSynthesizer(
        model="bulbul:v3",
        voice_id="v",
        language="hi-IN",
        synthesizer_key="k",
        task_manager_instance=MagicMock(),
        expressive_tags=True,
    )
    pool = SynthesizerPool(
        synthesizers={"en": _voice(ElevenlabsV3Synthesizer, "eleven_v4_turbo"), "hi": hindi},
        active_label="en",
        multilingual_config={},
    )
    assert "<<warm>>" in _tm(True, pool)._TaskManager__with_expressive_tags_note("")


def test_no_guide_when_a_language_voice_would_speak_tags_aloud():
    unhandled = MagicMock(spec=["synthesize", "push"])  # a voice outside the streaming path, with no tag step
    pool = SynthesizerPool(
        synthesizers={"en": _voice(ElevenlabsV3Synthesizer, "eleven_v4_turbo"), "hi": unhandled},
        active_label="en",
        multilingual_config={},
    )
    assert _tm(True, pool)._TaskManager__with_expressive_tags_note("") == ""


def test_a_replayed_reply_is_billed_for_its_spoken_words_only():
    assert _tm(True)._TaskManager__billed_characters(REPLY) == len(
        "Sorry for the wait. Your booking BK4821 is confirmed."
    )
    assert _tm(False)._TaskManager__billed_characters(REPLY) == len(REPLY)


async def test_a_line_rendered_ahead_of_time_with_tags_is_spoken_live():
    import time as _time
    from unittest.mock import AsyncMock

    synth = MagicMock(get_engine=MagicMock(return_value="sonic-3.6"), push=AsyncMock())
    tm = _tm(True, synth)
    tm.turn_based_conversation, tm.conversation_ended = False, False
    tm.conversation_start_init_ts, tm.synthesizer_provider, tm.run_id = _time.time() * 1000, "cartesia", "run"
    tm.interruption_manager = MagicMock(is_valid_sequence=MagicMock(return_value=True))
    tm.task_config = {"task_type": "conversation", "tools_config": {"output": {"provider": "plivo"}}}
    tm.tts_response_cache, tm.synthesizer_characters = None, 0
    tm._TaskManager__send_preprocessed_audio = AsyncMock()
    welcome = "<<warm>> Hello! [[pause 400]] How can I help?"
    meta = {
        "sequence_id": -1,
        "message_category": "agent_welcome_message",
        "cached": True,
        "is_md5_hash": False,
        "text": welcome,
        "is_first_message": True,
    }
    await tm._synthesize({"data": welcome, "meta_info": meta})
    tm._TaskManager__send_preprocessed_audio.assert_not_called()
    assert synth.push.await_args.args[0]["data"] == welcome
