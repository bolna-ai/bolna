"""Unit tests for FunASR / SenseVoice self-hosted STT provider."""

import asyncio
import audioop
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from bolna.enums import TranscriberProvider
from bolna.models import Transcriber
from bolna.providers import SUPPORTED_TRANSCRIBER_PROVIDERS
from bolna.transcriber.funasr_transcriber import FunASRTranscriber


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


@pytest.mark.asyncio
async def test_receiver_emits_interim_then_final():
    tr = FunASRTranscriber(telephony_provider="web_based_call", input_queue=asyncio.Queue(), output_queue=asyncio.Queue())
    tr.meta_info = {"request_id": "r1"}

    class FakeWS:
        def __init__(self, messages):
            self._messages = messages

        def __aiter__(self):
            async def _gen():
                for m in self._messages:
                    yield m

            return _gen()

    ws = FakeWS(
        [
            json.dumps({"mode": "2pass-online", "text": "hel"}),
            json.dumps({"mode": "2pass-offline", "text": "hello", "is_final": True}),
        ]
    )

    packets = []
    async for packet in tr.receiver(ws):
        packets.append(packet)

    types = []
    for p in packets:
        data = p["data"]
        if isinstance(data, str):
            types.append(data)
        else:
            types.append(data.get("type"))

    assert "speech_started" in types
    assert "interim_transcript_received" in types
    assert "transcript" in types
    finals = [p["data"]["content"] for p in packets if isinstance(p["data"], dict) and p["data"].get("type") == "transcript"]
    assert finals == ["hello"]


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
