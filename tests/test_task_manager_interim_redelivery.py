"""Deepgram can re-send the caller's just-finalized sentence as a fresh interim while the agent's
reply to it is playing. That repeat is not new speech and must not barge in: it cut the reply and,
since the repeat never finalizes, nothing regenerated and the agent went silent.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory

QUESTION = "what is today's special offer"
REPLY = "Twenty percent off."


def _interim(content):
    return {
        "data": {"type": "interim_transcript_received", "content": content},
        "meta_info": {"io": "freeswitch", "sequence_id": 2},
    }


def _history(reply_committed=True):
    history = ConversationHistory([{"role": "system", "content": "prompt"}])
    history.append_welcome_message("Hello Jordan Rivera, thanks for calling. How can I help?")
    # the transcriber hands the final over with a leading space
    history.append_user(f" {QUESTION}")
    if reply_committed:
        history.append_assistant(REPLY)
    return history


def _make_tm(*, history, audio_playing, response_in_pipeline=False, regen_armed=False, ms_since_final=1000):
    tm = MagicMock()
    tm.hangup_triggered = False
    tm._end_call_in_progress = False
    tm.function_call_in_flight = False
    tm.has_transfer = False
    tm.conversation_ended = False
    tm._hangup_interruptible_window = False
    tm.stream = True
    tm.response_in_pipeline = response_in_pipeline
    tm.regen_settle_armed = MagicMock(return_value=regen_armed)
    tm.transcriber_output_queue = asyncio.Queue()
    tm.process_transcriber_request = AsyncMock(return_value=0)
    tm._set_call_details = MagicMock()
    tm._get_next_step = MagicMock(return_value="llm")
    tm.tools = {"input": MagicMock(), "transcriber": MagicMock()}
    tm.tools["input"].welcome_message_played = MagicMock(return_value=True)
    tm.tools["input"].is_audio_being_played_to_user = MagicMock(return_value=audio_playing)
    tm.conversation_history = history
    # the threshold is met: this is the call that cut the agent off in production
    tm.interruption_manager.should_trigger_interruption = MagicMock(return_value=True)
    # production: the repeat landed 1.0s after the final
    tm.interruption_manager.get_time_since_utterance_end = MagicMock(return_value=ms_since_final)
    tm._TaskManager__cleanup_downstream_tasks = AsyncMock()
    tm._end_call_on_component_error = AsyncMock()
    tm.task_config = {"tools_config": {"transcriber": {"provider": "deepgram"}}}
    tm._should_ignore_transcriber_input = TaskManager._should_ignore_transcriber_input.__get__(tm, TaskManager)
    tm._listen_transcriber = TaskManager._listen_transcriber.__get__(tm, TaskManager)
    return tm


async def _drive(tm, *contents):
    for content in contents:
        await tm.transcriber_output_queue.put(_interim(content))
    try:
        await asyncio.wait_for(tm._listen_transcriber(), timeout=0.3)
    except asyncio.TimeoutError:
        pass


def _barged_in(tm):
    return tm._TaskManager__cleanup_downstream_tasks.await_count > 0


async def test_repeat_of_answered_question_while_reply_plays_does_not_barge_in():
    # the production sequence: reply committed and playing, response no longer "in pipeline"
    tm = _make_tm(history=_history(), audio_playing=True)
    await _drive(tm, QUESTION)
    assert not _barged_in(tm)
    tm.interruption_manager.should_trigger_interruption.assert_not_called()


async def test_new_words_while_reply_plays_still_barge_in():
    tm = _make_tm(history=_history(), audio_playing=True)
    await _drive(tm, "no wait I meant tomorrow")
    assert _barged_in(tm)


async def test_repeat_then_real_speech_still_barges_in():
    # skipping the phantom must not swallow genuine speech that follows it
    tm = _make_tm(history=_history(), audio_playing=True)
    await _drive(tm, QUESTION, "actually cancel my order")
    assert _barged_in(tm)


async def test_repeat_long_after_the_final_is_real_speech_and_barges_in():
    # outside the re-delivery window a word-for-word repeat is the caller saying it again
    tm = _make_tm(history=_history(), audio_playing=True, ms_since_final=3000)
    await _drive(tm, QUESTION)
    assert _barged_in(tm)


async def test_repeat_with_no_final_on_record_barges_in():
    # -1: utterance end was reset (e.g. user continuation), so there is no final to be a re-delivery of
    tm = _make_tm(history=_history(), audio_playing=True, ms_since_final=-1)
    await _drive(tm, QUESTION)
    assert _barged_in(tm)


async def test_repeat_after_reply_finished_is_left_to_the_normal_path():
    # nothing playing: a repeated question is a real new turn, so the guard must not skip it
    tm = _make_tm(history=_history(), audio_playing=False)
    await _drive(tm, QUESTION)
    tm.interruption_manager.should_trigger_interruption.assert_called_once()


async def test_existing_guard_before_first_audio_still_skips():
    # reply generating, nothing committed or playing yet: the original late-delivery case
    tm = _make_tm(history=_history(reply_committed=False), audio_playing=False, response_in_pipeline=True)
    await _drive(tm, QUESTION)
    assert not _barged_in(tm)
    tm.interruption_manager.should_trigger_interruption.assert_not_called()


class TestRepeatsLastUserTurn:
    def test_matches_last_user_turn_past_the_reply(self):
        assert _history().repeats_last_user_turn(QUESTION) is True

    def test_is_duplicate_user_still_false_once_reply_committed(self):
        # the final-transcript path relies on this: a re-asked question after the answer is a new turn
        assert _history().is_duplicate_user(QUESTION) is False

    def test_different_words_do_not_match(self):
        assert _history().repeats_last_user_turn("what is tomorrow's special offer") is False

    def test_only_the_most_recent_user_turn_counts(self):
        history = _history()
        history.append_user(" thanks")
        assert history.repeats_last_user_turn(QUESTION) is False
        assert history.repeats_last_user_turn("thanks") is True

    def test_no_user_turn_yet(self):
        history = ConversationHistory([{"role": "system", "content": "prompt"}])
        history.append_welcome_message("Hello")
        assert history.repeats_last_user_turn("Hello") is False

    def test_user_row_without_content(self):
        history = ConversationHistory([{"role": "system", "content": "prompt"}])
        history.append_user(None)
        assert history.repeats_last_user_turn(QUESTION) is False
