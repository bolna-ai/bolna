from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("PLIVO_AUTH_ID", "test")
os.environ.setdefault("PLIVO_AUTH_TOKEN", "test")

from bolna.input_handlers.telephony import TelephonyInputHandler
from bolna.input_handlers.telephony_providers.exotel import ExotelInputHandler
from bolna.input_handlers.telephony_providers.plivo import PlivoInputHandler
from bolna.input_handlers.telephony_providers.twilio import TwilioInputHandler
from bolna.input_handlers.telephony_providers.vobiz import VobizInputHandler


def _common_kwargs(speech_events_queue=None, vad_config=None):
    return dict(
        queues={
            "transcriber": asyncio.Queue(),
            "dtmf": asyncio.Queue(),
        },
        websocket=None,
        input_types={"audio": 1},
        mark_event_meta_data=None,
        turn_based_conversation=False,
        is_welcome_message_played=False,
        observable_variables={},
        vad_config=vad_config,
        speech_events_queue=speech_events_queue,
    )


PROVIDER_CLASSES = [
    ("twilio", TwilioInputHandler),
    ("plivo", PlivoInputHandler),
    ("vobiz", VobizInputHandler),
    ("exotel", ExotelInputHandler),
]


def _instantiate(cls, **kwargs):
    # PlivoInputHandler instantiates a RestClient in __init__ which validates creds.
    if cls is PlivoInputHandler:
        with patch("bolna.input_handlers.telephony_providers.plivo.plivosdk.RestClient"):
            return cls(**kwargs)
    return cls(**kwargs)


@pytest.mark.parametrize("io_provider,cls", PROVIDER_CLASSES)
def test_subclass_accepts_vad_kwargs_without_typeerror(io_provider, cls):
    queue = asyncio.Queue()
    handler = _instantiate(cls, **_common_kwargs(speech_events_queue=queue, vad_config=None))
    assert handler.io_provider == io_provider
    assert handler._turn_detector is None
    assert handler._speech_events_queue is queue


@pytest.mark.parametrize("io_provider,cls", PROVIDER_CLASSES)
def test_subclass_builds_turn_detector_when_vad_enabled(io_provider, cls):
    queue = asyncio.Queue()
    cfg = {
        "sample_rate": 8000,
        "threshold": 0.5,
        "min_silence_ms": 100,
        "speech_pad_ms": 30,
        "pre_speech_ms": 500,
    }
    handler = _instantiate(cls, **_common_kwargs(speech_events_queue=queue, vad_config=cfg))
    assert handler.io_provider == io_provider
    assert handler._turn_detector is not None
    # The detector should report not-yet-in-speech and respect the configured
    # 500ms at 8kHz mulaw == 4000 bytes
    assert handler._turn_detector.is_in_speech is False
    assert handler._turn_detector.pre_speech_buffer.capacity_bytes == 4000


def test_base_class_accepts_kwargs_too():
    handler = TelephonyInputHandler(**_common_kwargs(vad_config=None))
    assert handler._turn_detector is None
