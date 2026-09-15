"""DTMF on the FreeSWITCH fork: digits from {"type":"dtmf"} text frames collect into one entry."""

import asyncio
from unittest.mock import MagicMock

import pytest

from bolna.enums import TelephonyProvider
from bolna.input_handlers import dtmf as dtmf_module
from bolna.input_handlers.telephony_providers.freeswitch import FreeSwitchInputHandler
from bolna.models import IOModel


@pytest.fixture(autouse=True)
def fast_interdigit(monkeypatch):
    monkeypatch.setattr(dtmf_module, "DTMF_INTERDIGIT_TIMEOUT_S", 0.05)


def _make_handler(active=True):
    handler = FreeSwitchInputHandler(queues={"dtmf": asyncio.Queue(), "transcriber": asyncio.Queue()})
    handler.is_dtmf_active = active
    handler.on_playout_done = MagicMock()
    return handler


def test_trunk_is_a_label_not_an_audio_transport():
    assert TelephonyProvider.TRUNK.value == "trunk"
    assert "trunk" in TelephonyProvider.all_values()
    # membership in telephony_values() would force 8 kHz ASR; the fork is 16 kHz linear16
    assert "trunk" not in TelephonyProvider.telephony_values()
    assert "trunk" not in TelephonyProvider.mulaw_values()
    assert "freeswitch" not in TelephonyProvider.telephony_values()


def test_trunk_audio_config_validates():
    assert IOModel(provider="freeswitch", format="linear16").provider == "freeswitch"


async def test_digits_within_timeout_collect_into_one_entry():
    handler = _make_handler()
    await handler.process_message({"type": "dtmf", "digit": "5", "duration_ms": 160})
    await handler.process_message({"type": "dtmf", "digit": "5", "duration_ms": 160})
    assert handler.queues["dtmf"].empty()
    await asyncio.sleep(0.15)
    assert handler.queues["dtmf"].get_nowait() == "55"
    assert handler.queues["dtmf"].empty()


async def test_terminator_submits_immediately_without_itself():
    handler = _make_handler()
    await handler.process_message({"type": "dtmf", "digit": "1"})
    await handler.process_message({"type": "dtmf", "digit": "#"})
    assert handler.queues["dtmf"].get_nowait() == "1"


async def test_dtmf_ignored_when_agent_has_dtmf_disabled():
    handler = _make_handler(active=False)
    await handler.process_message({"type": "dtmf", "digit": "5"})
    await asyncio.sleep(0.15)
    assert handler.queues["dtmf"].empty()


async def test_playout_done_still_routed():
    handler = _make_handler()
    await handler.process_message({"type": "playoutDone"})
    handler.on_playout_done.assert_called_once()
