"""The per-call Deepgram host override must reach every label in the multilingual transcriber pool."""

from unittest.mock import MagicMock

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager


def _run_setup(monkeypatch, transcriber_config):
    """Drive the real __setup_transcriber multilingual branch against a mock TaskManager.

    The branch mutates each per-label cfg dict in place before spreading it into the
    transcriber, so we assert on the mutated multilingual dicts directly.
    """
    monkeypatch.setattr(tmmod, "SUPPORTED_TRANSCRIBER_PROVIDERS", {"deepgram": MagicMock()})
    monkeypatch.setattr(tmmod, "TranscriberPool", MagicMock())

    tm = MagicMock()
    tm.task_config = {"tools_config": {"transcriber": transcriber_config}}
    tm.turn_based_conversation = True  # provider='playground'; skips input-provider lookup
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.kwargs = {}
    tm._TaskManager__language_switch_enabled = MagicMock(return_value=False)
    tm._TaskManager__setup_transcriber = TaskManager._TaskManager__setup_transcriber.__get__(tm, TaskManager)

    tm._TaskManager__setup_transcriber()


def test_override_propagates_to_every_label(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "deepgram_host": "self-hosted:8080",
        "deepgram_flux_host": "self-hosted-flux:8080",
        "deepgram_host_protocol": "ws",
        "multilingual": {
            "en": {"provider": "deepgram", "model": "nova-2"},
            "hi": {"provider": "deepgram", "model": "flux-general-hi"},
        },
    }
    _run_setup(monkeypatch, cfg)

    for label in ("en", "hi"):
        per_label = cfg["multilingual"][label]
        assert per_label["deepgram_host"] == "self-hosted:8080"
        assert per_label["deepgram_flux_host"] == "self-hosted-flux:8080"
        assert per_label["deepgram_host_protocol"] == "ws"


def test_per_label_override_wins(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "deepgram_host": "top-level:8080",
        "deepgram_host_protocol": "ws",
        "multilingual": {
            "en": {"provider": "deepgram", "model": "nova-2", "deepgram_host": "en-specific:9090"},
            "hi": {"provider": "deepgram", "model": "nova-2"},
        },
    }
    _run_setup(monkeypatch, cfg)

    assert cfg["multilingual"]["en"]["deepgram_host"] == "en-specific:9090"
    assert cfg["multilingual"]["hi"]["deepgram_host"] == "top-level:8080"
    # Protocol has no per-label value, so both inherit the top-level one.
    assert cfg["multilingual"]["en"]["deepgram_host_protocol"] == "ws"
    assert cfg["multilingual"]["hi"]["deepgram_host_protocol"] == "ws"


def test_no_override_leaves_labels_untouched(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "multilingual": {"en": {"provider": "deepgram", "model": "nova-2"}},
    }
    _run_setup(monkeypatch, cfg)

    assert "deepgram_host" not in cfg["multilingual"]["en"]
    assert "deepgram_host_protocol" not in cfg["multilingual"]["en"]
