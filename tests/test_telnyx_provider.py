"""Telnyx Media Streaming (https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming)
follows the same start/media/mark/stop shape Twilio uses, but with two deviations this
covers: the call id lives at start.call_control_id (not start.callSid) with the stream id
at the top level as stream_id (not nested), and outbound media/mark/clear frames carry no
session id at all - the websocket connection alone identifies the stream.
"""

import asyncio
import base64

import audioop
import pytest

from bolna.enums import TelephonyProvider
from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.input_handlers.telephony_providers.telnyx import TelnyxInputHandler
from bolna.output_handlers.telephony_providers.telnyx import TelnyxOutputHandler
from bolna.providers import (
    SUPPORTED_INPUT_HANDLERS,
    SUPPORTED_INPUT_TELEPHONY_HANDLERS,
    SUPPORTED_OUTPUT_HANDLERS,
    SUPPORTED_OUTPUT_TELEPHONY_HANDLERS,
)


def test_telnyx_registered_as_a_mulaw_telephony_provider():
    assert TelephonyProvider.TELNYX in TelephonyProvider.telephony_providers()
    assert TelephonyProvider.TELNYX in TelephonyProvider.mulaw_providers()
    assert SUPPORTED_INPUT_HANDLERS["telnyx"] is TelnyxInputHandler
    assert SUPPORTED_OUTPUT_HANDLERS["telnyx"] is TelnyxOutputHandler
    assert SUPPORTED_INPUT_TELEPHONY_HANDLERS["telnyx"] is TelnyxInputHandler
    assert SUPPORTED_OUTPUT_TELEPHONY_HANDLERS["telnyx"] is TelnyxOutputHandler


class TestTelnyxInputHandler:
    def _make_handler(self):
        return TelnyxInputHandler(queues={"transcriber": asyncio.Queue()})

    @pytest.mark.asyncio
    async def test_call_start_reads_call_control_id_and_top_level_stream_id(self):
        handler = self._make_handler()
        packet = {
            "event": "start",
            "stream_id": "8f9e6de0-6d67-4c2e-8f9a-3d0f3a5f6b1a",
            "start": {
                "call_control_id": "v3:abc123",
                "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
            },
        }

        await handler.call_start(packet)

        assert handler.call_sid == "v3:abc123"
        assert handler.stream_sid == "8f9e6de0-6d67-4c2e-8f9a-3d0f3a5f6b1a"

    def test_mark_event_lookup_reads_mark_name(self):
        handler = self._make_handler()
        handler.mark_event_meta_data = MarkEventMetaData()
        handler.mark_event_meta_data.update_data("mark-1", {"text_synthesized": "hello"})

        data = handler.get_mark_event_meta_data_obj({"event": "mark", "mark": {"name": "mark-1"}})

        assert data["text_synthesized"] == "hello"


class TestTelnyxOutputHandler:
    def _make_handler(self):
        return TelnyxOutputHandler(mark_event_meta_data=MarkEventMetaData())

    @pytest.mark.asyncio
    async def test_media_message_has_no_session_id_and_converts_to_mulaw(self):
        handler = self._make_handler()
        pcm_audio = b"\x10\x00\x20\x00"

        message = await handler.form_media_message(pcm_audio, audio_format="pcm")

        assert message == {
            "event": "media",
            "media": {"payload": base64.b64encode(audioop.lin2ulaw(pcm_audio, 2)).decode("utf-8")},
        }

    @pytest.mark.asyncio
    async def test_media_message_skips_conversion_when_already_mulaw(self):
        handler = self._make_handler()
        mulaw_audio = b"\xff\x7e\x81"

        message = await handler.form_media_message(mulaw_audio, audio_format="mulaw")

        assert message["media"]["payload"] == base64.b64encode(mulaw_audio).decode("utf-8")

    @pytest.mark.asyncio
    async def test_mark_message_has_no_session_id(self):
        handler = self._make_handler()

        message = await handler.form_mark_message("mark-1")

        assert message == {"event": "mark", "mark": {"name": "mark-1"}}

    @pytest.mark.asyncio
    async def test_interruption_sends_a_bare_clear_event(self):
        sent = []

        class _FakeWebSocket:
            async def send_text(self, text):
                sent.append(text)

        handler = self._make_handler()
        handler.websocket = _FakeWebSocket()

        await handler.handle_interruption()

        assert sent == ['{"event": "clear"}']

    @pytest.mark.asyncio
    async def test_interruption_after_close_is_a_noop(self):
        handler = self._make_handler()
        handler._closed = True
        handler.websocket = None  # would AttributeError if handle_interruption tried to use it

        await handler.handle_interruption()
