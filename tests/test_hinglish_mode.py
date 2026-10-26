import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from bolna.models import Transcriber
from bolna.transcriber.assemblyai_transcriber import AssemblyAITranscriber
from bolna.transcriber.deepgram_transcriber import DeepgramTranscriber
from bolna.transcriber.sarvam_transcriber import SarvamTranscriber
from bolna.transcriber.soniox_transcriber import SonioxTranscriber


def _query(url):
    return parse_qs(urlparse(url).query)


@pytest.mark.parametrize(
    "kwargs, hints, strict",
    [
        ({"language": "hi", "hinglish_mode": True}, ["hi", "en"], True),
        ({"language": "hi"}, ["hi"], None),
        ({"language": "hi", "hinglish_mode": False}, ["hi"], None),
        ({"language": "en", "hinglish_mode": True}, ["en"], None),
        ({"language": "hi", "hinglish_mode": True, "language_hints": ["hi", "pa"]}, ["hi", "pa"], None),
    ],
)
def test_soniox_sends_strict_hindi_english_hints(kwargs, hints, strict):
    config = SonioxTranscriber("plivo", output_queue=asyncio.Queue(), **kwargs)._build_config()
    assert config["language_hints"] == hints
    assert config.get("language_hints_strict") == strict


@pytest.mark.parametrize(
    "model, language, hinglish_mode, mode",
    [
        ("saaras:v3", "hi-IN", True, ["codemix"]),
        ("saaras:v4", "hi-IN", True, ["codemix"]),
        ("saaras:v3", "hi-IN", False, ["transcribe"]),
        ("saaras:v3", "ta-IN", True, ["transcribe"]),
        ("saarika:v2.5", "hi-IN", True, None),
    ],
)
def test_sarvam_uses_codemix_mode(model, language, hinglish_mode, mode):
    transcriber = SarvamTranscriber(
        telephony_provider="twilio",
        model=model,
        language=language,
        stream=True,
        transcriber_key="test-key",
        hinglish_mode=hinglish_mode,
    )
    assert _query(transcriber.ws_url).get("mode") == mode


@pytest.mark.parametrize(
    "model, language, hinglish_mode, codes",
    [
        ("universal-3-6-pro", "hi", True, '["hi", "en"]'),
        ("universal-3-6-pro", "hi", False, '["hi"]'),
        ("universal-3-6-pro", "en", True, '["en"]'),
        ("universal-3-5-pro", "hi", True, '["hi"]'),
    ],
)
def test_assemblyai_sends_hindi_and_english_codes(model, language, hinglish_mode, codes):
    transcriber = AssemblyAITranscriber(
        "twilio", model=model, language=language, input_queue=asyncio.Queue(), hinglish_mode=hinglish_mode
    )
    assert _query(transcriber.get_assemblyai_ws_url())["language_codes"] == [codes]


@pytest.mark.parametrize(
    "language, language_hints, hinglish_mode, hints",
    [
        ("hi", None, True, ["hi", "en"]),
        ("multi", ["hi"], True, ["hi", "en"]),
        ("hi", None, False, ["hi"]),
        ("ta", None, True, ["ta"]),
    ],
)
def test_deepgram_flux_multi_adds_english_hint(language, language_hints, hinglish_mode, hints):
    transcriber = DeepgramTranscriber(
        telephony_provider="plivo",
        model="flux-general-multi",
        language=language,
        stream=True,
        language_hints=language_hints,
        hinglish_mode=hinglish_mode,
    )
    assert _query(transcriber._get_flux_ws_url())["language_hint"] == hints


def test_saved_agent_keeps_hinglish_mode():
    assert Transcriber(provider="soniox", language="hi", hinglish_mode=True).hinglish_mode is True
