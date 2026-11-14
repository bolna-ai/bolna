"""Vobiz streams linear16 at 8 kHz, like Plivo and Exotel; every transcriber must declare 8 kHz for it."""

import asyncio

import pytest

from bolna.transcriber.gladia_transcriber import GladiaTranscriber
from bolna.transcriber.pixa_transcriber import PixaTranscriber
from bolna.transcriber.smallest_transcriber import SmallestTranscriber


def _build(cls, provider):
    # task_manager passes the saved Transcriber config, whose sampling_rate defaults to 16000.
    return cls(
        provider,
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        transcriber_key="placeholder",
        encoding="linear16",
        sampling_rate=16000,
    )


def _declared_rate(transcriber):
    return getattr(transcriber, "sample_rate", None) or transcriber.sampling_rate


@pytest.mark.parametrize("cls", [GladiaTranscriber, SmallestTranscriber, PixaTranscriber])
async def test_vobiz_audio_is_declared_as_8k_linear16_like_plivo(cls):
    plivo = _build(cls, "plivo")
    vobiz = _build(cls, "vobiz")

    assert _declared_rate(plivo) == 8000
    assert (vobiz.encoding, _declared_rate(vobiz)) == (plivo.encoding, 8000)
