"""A BYOK key reaches only the multilingual legs whose provider it was resolved for.

dashboard-backend resolves one ``transcriber_key`` / ``synthesizer_key`` from the *top-level*
provider. Handing that key to every leg sent a customer's Deepgram key to a Sarvam leg, which
403'd on every connect and hung up the call on a language switch. A leg on a different provider
must get no key, so it falls back to the platform env key.
"""

from unittest.mock import MagicMock

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager


def _run_transcriber_setup(monkeypatch, transcriber_config, kwargs):
    deepgram_cls, sarvam_cls = MagicMock(), MagicMock()
    monkeypatch.setattr(tmmod, "SUPPORTED_TRANSCRIBER_PROVIDERS", {"deepgram": deepgram_cls, "sarvam": sarvam_cls})
    monkeypatch.setattr(tmmod, "TranscriberPool", MagicMock())

    tm = MagicMock()
    tm.task_config = {"tools_config": {"transcriber": transcriber_config}}
    tm.turn_based_conversation = True
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.kwargs = kwargs
    tm._TaskManager__language_switch_enabled = MagicMock(return_value=False)
    tm._TaskManager__setup_transcriber = TaskManager._TaskManager__setup_transcriber.__get__(tm, TaskManager)

    tm._TaskManager__setup_transcriber()
    return deepgram_cls, sarvam_cls


def _run_synthesizer_setup(monkeypatch, synthesizer_config, kwargs):
    elevenlabs_cls, cartesia_cls = MagicMock(), MagicMock()
    monkeypatch.setattr(tmmod, "SUPPORTED_SYNTHESIZER_MODELS", {"elevenlabs": elevenlabs_cls, "cartesia": cartesia_cls})
    monkeypatch.setattr(tmmod, "SynthesizerPool", MagicMock())

    tm = MagicMock()
    tm.task_config = {
        "tools_config": {"synthesizer": synthesizer_config, "output": {"provider": "vobiz"}, "llm_agent": None}
    }
    tm._is_conversation_task = MagicMock(return_value=False)
    tm.turn_based_conversation = False
    tm.is_web_based_call = False
    tm.language_switcher = None
    tm.kwargs = kwargs
    tm._TaskManager__setup_synthesizer = TaskManager._TaskManager__setup_synthesizer.__get__(tm, TaskManager)

    tm._TaskManager__setup_synthesizer()
    return elevenlabs_cls, cartesia_cls


def _mixed_transcriber_config():
    return {
        "provider": "deepgram",
        "multilingual": {
            "en": {"provider": "deepgram", "model": "nova-3", "language": "en"},
            "kn": {"provider": "sarvam", "model": "saaras:v3", "language": "kn"},
        },
        "active": "en",
    }


def _mixed_synthesizer_config():
    return {
        "provider": "elevenlabs",
        "multilingual": {
            "en": {"provider": "elevenlabs", "provider_config": {"voice_id": "v1"}},
            "kn": {"provider": "cartesia", "provider_config": {"voice_id": "v2", "language": "kn"}},
        },
        "active": "en",
    }


def test_transcriber_key_only_reaches_matching_provider_leg(monkeypatch):
    deepgram_cls, sarvam_cls = _run_transcriber_setup(
        monkeypatch, _mixed_transcriber_config(), {"transcriber_key": "dg-byok", "run_id": "r1"}
    )

    assert deepgram_cls.call_args.kwargs["transcriber_key"] == "dg-byok"
    assert "transcriber_key" not in sarvam_cls.call_args.kwargs
    assert sarvam_cls.call_args.kwargs["run_id"] == "r1"


def test_synthesizer_key_only_reaches_matching_provider_leg(monkeypatch):
    elevenlabs_cls, cartesia_cls = _run_synthesizer_setup(
        monkeypatch, _mixed_synthesizer_config(), {"synthesizer_key": "el-byok", "run_id": "r1"}
    )

    assert elevenlabs_cls.call_args.kwargs["synthesizer_key"] == "el-byok"
    assert "synthesizer_key" not in cartesia_cls.call_args.kwargs
    assert cartesia_cls.call_args.kwargs["run_id"] == "r1"


def test_setup_does_not_strip_keys_from_shared_kwargs(monkeypatch):
    kwargs = {"transcriber_key": "dg-byok", "synthesizer_key": "el-byok"}
    _run_transcriber_setup(monkeypatch, _mixed_transcriber_config(), kwargs)
    _run_synthesizer_setup(monkeypatch, _mixed_synthesizer_config(), kwargs)

    assert kwargs == {"transcriber_key": "dg-byok", "synthesizer_key": "el-byok"}
