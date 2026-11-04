"""Unit tests for FunASR / SenseVoice self-hosted STT provider."""

import asyncio
import audioop
import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from bolna.enums import TranscriberProvider
from bolna.models import Transcriber
from bolna.providers import SUPPORTED_TRANSCRIBER_PROVIDERS
from bolna.transcriber.funasr_transcriber import FunASRTranscriber


class FakeWS:
    def __init__(self, messages):
        self._messages = messages
        self.sent = []

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m

        return _gen()

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        return None


def _packet_types(packets):
    types = []
    for p in packets:
        data = p["data"]
        if isinstance(data, str):
            types.append(data)
        else:
            types.append(data.get("type"))
    return types


def test_funasr_provider_registered():
    assert TranscriberProvider.FUNASR.value == "funasr"
    assert "funasr" in TranscriberProvider.all_values()
    assert SUPPORTED_TRANSCRIBER_PROVIDERS["funasr"] is FunASRTranscriber


def test_transcriber_model_accepts_funasr():
    t = Transcriber(provider="funasr", model="sensevoice", stream=True, language="en")
    assert t.provider == "funasr"
    assert t.model == "sensevoice"


def test_transcriber_model_rejects_unknown_provider():
    with pytest.raises(Exception):
        Transcriber(provider="not-a-real-asr", model="x", stream=True)


def test_default_protocol_is_realtime():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
    )
    assert tr.protocol == "realtime"


def test_classic_wss_protocol_can_be_selected():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        funasr_protocol="wss",
    )
    assert tr.protocol == "wss"


def test_to_pcm16k_mulaw_8k():
    # 20ms of silence at 8k mulaw
    pcm8 = np.zeros(160, dtype=np.int16).tobytes()
    mulaw = audioop.lin2ulaw(pcm8, 2)
    tr = FunASRTranscriber(telephony_provider="twilio", input_queue=asyncio.Queue(), output_queue=asyncio.Queue())
    assert tr.encoding == "mulaw"
    assert tr.sampling_rate == 8000
    out = tr._to_pcm16k(mulaw)
    assert len(out) > 0
    assert len(out) % 2 == 0
    # 8k → 16k roughly doubles sample count
    assert abs(len(out) - len(pcm8) * 2) <= 4


def test_build_wss_config_includes_mode_and_sample_rate():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        keywords="bolna,voice",
    )
    cfg = tr._build_wss_config()
    assert cfg["mode"] == "2pass"
    assert cfg["audio_fs"] == 16000
    assert cfg["wav_format"] == "pcm"
    assert cfg["is_speaking"] is True
    assert "bolna" in cfg["hotwords"]


def test_is_final_and_interim_modes():
    tr = FunASRTranscriber(telephony_provider="web_based_call", input_queue=asyncio.Queue(), output_queue=asyncio.Queue())
    assert tr._is_final_message({"mode": "2pass-offline", "text": "hi"})
    assert tr._is_final_message({"is_final": True, "text": "hi"})
    assert tr._is_interim_message({"mode": "2pass-online", "text": "h"})
    assert tr._is_interim_message({"is_final": False, "text": "h"})


def test_non_stream_default_uses_funasr_server_http_port():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        stream=False,
    )
    assert tr.ws_url == "ws://127.0.0.1:10095"
    assert tr.http_base_url == "http://127.0.0.1:8000"


def test_non_stream_http_base_url_can_be_overridden():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        stream=False,
        http_base_url="http://localhost:18000",
    )
    assert tr.http_base_url == "http://localhost:18000"


@pytest.mark.asyncio
async def test_receiver_emits_interim_then_final():
    tr = FunASRTranscriber(telephony_provider="web_based_call", input_queue=asyncio.Queue(), output_queue=asyncio.Queue())
    tr.meta_info = {"request_id": "r1"}

    ws = FakeWS(
        [
            json.dumps({"mode": "2pass-online", "text": "hel"}),
            json.dumps({"mode": "2pass-offline", "text": "hello", "is_final": True}),
        ]
    )

    packets = []
    async for packet in tr.receiver(ws):
        packets.append(packet)

    types = _packet_types(packets)
    assert "speech_started" in types
    assert "interim_transcript_received" in types
    assert "transcript" in types
    finals = [p["data"]["content"] for p in packets if isinstance(p["data"], dict) and p["data"].get("type") == "transcript"]
    assert finals == ["hello"]


@pytest.mark.asyncio
async def test_receiver_empty_final_emits_speech_ended():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        funasr_protocol="realtime",
    )
    tr.meta_info = {"request_id": "r1"}
    tr._speech_active = True

    ws = FakeWS([json.dumps({"is_final": True, "text": ""})])
    packets = []
    async for packet in tr.receiver(ws):
        packets.append(packet)

    assert _packet_types(packets) == ["speech_ended"]
    assert tr._speech_active is False


@pytest.mark.asyncio
async def test_realtime_partial_final_then_empty_close_turn():
    """Agent-loop contract: interim → final transcript; empty final closes turn."""
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        funasr_protocol="realtime",
    )
    tr.meta_info = {"request_id": "r1"}

    ws = FakeWS(
        [
            json.dumps({"is_final": False, "text": "hel"}),
            json.dumps({"is_final": True, "text": "hello"}),
            json.dumps({"is_final": True, "text": ""}),
        ]
    )
    packets = []
    async for packet in tr.receiver(ws):
        packets.append(packet)

    types = _packet_types(packets)
    assert types == ["speech_started", "interim_transcript_received", "transcript"]
    # Empty final after turn already reset is a no-op (no duplicate speech_ended).
    assert "speech_ended" not in types


@pytest.mark.asyncio
async def test_realtime_connect_sends_start():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        funasr_protocol="realtime",
    )
    mock_ws = AsyncMock()
    with patch("bolna.transcriber.funasr_transcriber.websockets.connect", new=AsyncMock(return_value=mock_ws)):
        with patch("bolna.transcriber.funasr_transcriber.get_ssl_context", return_value=None):
            ws = await tr.funasr_connect()
    assert ws is mock_ws
    mock_ws.send.assert_awaited()
    assert mock_ws.send.await_args_list[0].args[0] == "START"


@pytest.mark.asyncio
async def test_realtime_eos_sends_commit_then_stop():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        funasr_protocol="realtime",
        endpointing=50,
    )
    tr.meta_info = {}
    tr._speech_active = True
    ws = FakeWS([])
    await tr.input_queue.put({"data": b"", "meta_info": {"eos": True}})
    await tr.sender_stream(ws)
    assert ws.sent == ["COMMIT", "STOP"]


@pytest.mark.asyncio
async def test_realtime_silence_commit_after_audio():
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        funasr_protocol="realtime",
        endpointing=50,
    )
    pcm = np.zeros(320, dtype=np.int16).tobytes()
    ws = FakeWS([])
    await tr.input_queue.put({"data": pcm, "meta_info": {"request_id": "r1"}})
    await tr.input_queue.put({"data": b"", "meta_info": {"eos": True}})

    send_task = asyncio.create_task(tr.sender_stream(ws))
    await asyncio.wait_for(send_task, timeout=2.0)

    assert any(isinstance(m, (bytes, bytearray)) for m in ws.sent)
    assert "COMMIT" in ws.sent
    assert ws.sent[-1] == "STOP"


@pytest.mark.asyncio
async def test_connect_failure_emits_transcriber_connection_closed():
    out_q = asyncio.Queue()
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=out_q,
        funasr_protocol="realtime",
    )
    with patch.object(tr, "funasr_connect", new=AsyncMock(side_effect=ConnectionError("refused"))):
        await tr.transcribe()

    packet = out_q.get_nowait()
    assert packet["data"] == "transcriber_connection_closed"
    assert "refused" in (packet["meta_info"].get("connection_error") or "")


@pytest.mark.asyncio
async def test_http_batch_posts_multipart(monkeypatch):
    tr = FunASRTranscriber(
        telephony_provider="web_based_call",
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        stream=False,
        model="sensevoice",
        http_base_url="http://127.0.0.1:8000",
    )

    class FakeResp:
        status = 200

        async def text(self):
            return json.dumps({"text": "bonjour"})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeSession:
        def __init__(self):
            self.posted = None

        def post(self, url, data=None, headers=None, timeout=None):
            self.posted = {"url": url, "data": data, "headers": headers}
            return FakeResp()

        async def close(self):
            pass

    session = FakeSession()
    tr.http_session = session
    text = await tr._http_transcribe_batch(np.zeros(1600, dtype=np.int16).tobytes())
    assert text == "bonjour"
    assert session.posted["url"].endswith("/v1/audio/transcriptions")
