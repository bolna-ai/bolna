"""Per-label Deepgram host overrides reach the pool; the top-level host is NOT inherited into legs.

The router (dashboard-backend) stamps only the legs a chosen endpoint can actually serve, so an
unstamped leg must keep its default host rather than inherit a base host it may not support.
"""

from unittest.mock import MagicMock

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager


def _run_setup(monkeypatch, transcriber_config):
    """Drive the real __setup_transcriber multilingual branch against a mock TaskManager."""
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


def test_per_label_hosts_are_preserved(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "multilingual": {
            "en": {
                "provider": "deepgram",
                "model": "nova-3",
                "deepgram_host": "gcp:8080",
                "deepgram_host_protocol": "ws",
            },
            "hi": {
                "provider": "deepgram",
                "model": "nova-2",
                "deepgram_host": "modal:443",
                "deepgram_host_protocol": "wss",
            },
        },
    }
    _run_setup(monkeypatch, cfg)

    assert cfg["multilingual"]["en"]["deepgram_host"] == "gcp:8080"
    assert cfg["multilingual"]["en"]["deepgram_host_protocol"] == "ws"
    assert cfg["multilingual"]["hi"]["deepgram_host"] == "modal:443"
    assert cfg["multilingual"]["hi"]["deepgram_host_protocol"] == "wss"


def test_top_level_host_not_inherited_into_legs(monkeypatch):
    # An unstamped leg (no per-label host) must NOT pick up the top-level host — that inheritance is
    # exactly what would force an unsupported model onto the wrong endpoint.
    cfg = {
        "provider": "deepgram",
        "deepgram_host": "gcp:8080",
        "deepgram_host_protocol": "ws",
        "multilingual": {
            "en": {"provider": "deepgram", "model": "nova-3", "deepgram_host": "gcp:8080"},
            "hi": {"provider": "deepgram", "model": "flux-general-hi"},  # deliberately left on default
        },
    }
    _run_setup(monkeypatch, cfg)

    assert cfg["multilingual"]["en"]["deepgram_host"] == "gcp:8080"
    assert "deepgram_host" not in cfg["multilingual"]["hi"]
    assert "deepgram_host_protocol" not in cfg["multilingual"]["hi"]


def test_no_override_leaves_labels_untouched(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "multilingual": {"en": {"provider": "deepgram", "model": "nova-2"}},
    }
    _run_setup(monkeypatch, cfg)

    assert "deepgram_host" not in cfg["multilingual"]["en"]
    assert "deepgram_host_protocol" not in cfg["multilingual"]["en"]
