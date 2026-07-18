"""Unit tests for OpenAITranscriber speech_rms_threshold config and force-finalize (#711)."""

import asyncio
import json

import pytest

from bolna.constants import (
    OPENAI_TRANSCRIBER_SPEECH_RMS_THRESHOLD,
    OPENAI_TRANSCRIBER_TELEPHONY_SPEECH_RMS_THRESHOLD,
)
from bolna.models import Transcriber
from bolna.transcriber.openai_transcriber import OpenAITranscriber


def _make_transcriber(telephony_provider="default", **kwargs):
    return OpenAITranscriber(
        telephony_provider=telephony_provider,
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        model="gpt-realtime-whisper",
        language="en",
        **kwargs,
    )


def test_speech_rms_threshold_exposed_on_transcriber_model():
    cfg = Transcriber(provider="openai", speech_rms_threshold=250)
    assert cfg.speech_rms_threshold == 250


def test_speech_rms_threshold_default_none_on_model():
    cfg = Transcriber(provider="openai")
    assert cfg.speech_rms_threshold is None


def test_default_speech_rms_threshold_non_telephony():
    t = _make_transcriber(telephony_provider="default")
    assert t.speech_rms_threshold == OPENAI_TRANSCRIBER_SPEECH_RMS_THRESHOLD


def test_default_speech_rms_threshold_twilio():
    t = _make_transcriber(telephony_provider="twilio")
    assert t.speech_rms_threshold == OPENAI_TRANSCRIBER_TELEPHONY_SPEECH_RMS_THRESHOLD


@pytest.mark.parametrize("provider", ["plivo", "exotel", "vobiz"])
def test_default_speech_rms_threshold_other_telephony(provider):
    t = _make_transcriber(telephony_provider=provider)
    assert t.speech_rms_threshold == OPENAI_TRANSCRIBER_TELEPHONY_SPEECH_RMS_THRESHOLD


def test_explicit_speech_rms_threshold_overrides_telephony_default():
    t = _make_transcriber(telephony_provider="twilio", speech_rms_threshold=200)
    assert t.speech_rms_threshold == 200.0


def test_interim_transcript_text_joins_deltas():
    t = _make_transcriber()
    t.current_turn_interim_details = [
        {"transcript": "Hey ", "received_at": 1.0, "is_final": False},
        {"transcript": "Brooke", "received_at": 1.1, "is_final": False},
    ]
    assert t._interim_transcript_text() == "Hey Brooke"


@pytest.mark.asyncio
async def test_force_finalize_emits_transcript_from_interims():
    output_queue = asyncio.Queue()
    t = OpenAITranscriber(
        telephony_provider="twilio",
        input_queue=asyncio.Queue(),
        output_queue=output_queue,
        model="gpt-realtime-whisper",
    )
    t.current_turn_id = "turn_1"
    t._last_committed_turn_id = "turn_1"
    t._turn_committed = True
    t._commit_time = 0.0
    t.is_transcript_sent_for_processing = False
    t.current_turn_start_time = None
    t.current_turn_interim_details = [
        {"transcript": "Hey Brooke", "received_at": 1.0, "is_final": False},
    ]

    await t._force_finalize_from_interims()

    packet = output_queue.get_nowait()
    assert packet["data"]["type"] == "transcript"
    assert packet["data"]["content"] == "Hey Brooke"
    assert t._force_finalized_turn_id == "turn_1"
    assert t._final_transcript_event.is_set()
    assert any(entry.get("force_finalized") for entry in t.turn_latencies)


@pytest.mark.asyncio
async def test_force_finalize_without_interims_resets_without_transcript():
    output_queue = asyncio.Queue()
    t = OpenAITranscriber(
        telephony_provider="twilio",
        input_queue=asyncio.Queue(),
        output_queue=output_queue,
        model="gpt-realtime-whisper",
    )
    t.current_turn_id = "turn_2"
    t._last_committed_turn_id = "turn_2"
    t._turn_committed = True
    t.current_turn_interim_details = []

    await t._force_finalize_from_interims()

    assert output_queue.empty()
    assert t._force_finalized_turn_id is None
    assert t._final_transcript_event.is_set()


@pytest.mark.asyncio
async def test_receiver_ignores_late_completed_after_force_finalize():
    t = _make_transcriber(telephony_provider="twilio")
    t._force_finalized_turn_id = "turn_1"
    t._last_committed_turn_id = "turn_1"
    t.meta_info = {}

    class _FakeWS:
        def __init__(self, messages):
            self._messages = messages

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._messages:
                raise StopAsyncIteration
            return self._messages.pop(0)

    ws = _FakeWS(
        [
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Hey Brooke",
                    "item_id": "item_1",
                }
            )
        ]
    )

    packets = []
    async for packet in t.receiver(ws):
        packets.append(packet)

    assert packets == []
    assert t._force_finalized_turn_id is None
