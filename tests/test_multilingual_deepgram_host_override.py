"""Per-label Deepgram host overrides reach the pool; the top-level host is NOT inherited into legs.

The router (dashboard-backend) stamps only the legs a chosen endpoint can actually serve, so an
unstamped leg must keep its default host rather than inherit a base host it may not support.
"""

from unittest.mock import MagicMock

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager


def _run_setup(monkeypatch, transcriber_config):
    """Drive the real __setup_transcriber multilingual branch against a mock TaskManager; the config each leg
    is built with, by model."""
    factory = MagicMock()
    monkeypatch.setattr(tmmod, "SUPPORTED_TRANSCRIBER_PROVIDERS", {"deepgram": factory})
    monkeypatch.setattr(tmmod, "TranscriberPool", MagicMock())

    tm = MagicMock()
    tm.task_config = {"tools_config": {"transcriber": transcriber_config}}
    tm.turn_based_conversation = True  # provider='playground'; skips input-provider lookup
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.kwargs = {}
    tm._TaskManager__language_switch_enabled = MagicMock(return_value=False)
    tm._TaskManager__setup_transcriber = TaskManager._TaskManager__setup_transcriber.__get__(tm, TaskManager)
    tm._TaskManager__build_transcriber_leg = TaskManager._TaskManager__build_transcriber_leg.__get__(tm, TaskManager)
    tm._TaskManager__transcriber_fallback_entries = TaskManager._TaskManager__transcriber_fallback_entries.__get__(
        tm, TaskManager
    )

    tm._TaskManager__setup_transcriber()
    return {call.kwargs["model"]: call.kwargs for call in factory.call_args_list}


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
    legs = _run_setup(monkeypatch, cfg)

    assert legs["nova-3"]["deepgram_host"] == "gcp:8080"
    assert legs["nova-3"]["deepgram_host_protocol"] == "ws"
    assert legs["nova-2"]["deepgram_host"] == "modal:443"
    assert legs["nova-2"]["deepgram_host_protocol"] == "wss"


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
    legs = _run_setup(monkeypatch, cfg)

    assert legs["nova-3"]["deepgram_host"] == "gcp:8080"
    assert "deepgram_host" not in legs["flux-general-hi"]
    assert "deepgram_host_protocol" not in legs["flux-general-hi"]


def test_no_override_leaves_labels_untouched(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "multilingual": {"en": {"provider": "deepgram", "model": "nova-2"}},
    }
    legs = _run_setup(monkeypatch, cfg)

    assert "deepgram_host" not in legs["nova-2"]
    assert "deepgram_host_protocol" not in legs["nova-2"]


def test_top_level_mip_opt_out_inherited_into_legs(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "mip_opt_out": True,
        "multilingual": {
            "en": {"provider": "deepgram", "model": "nova-3"},
            "hi": {"provider": "deepgram", "model": "nova-2", "mip_opt_out": False},
        },
    }
    legs = _run_setup(monkeypatch, cfg)

    assert legs["nova-3"]["mip_opt_out"] is True
    assert legs["nova-2"]["mip_opt_out"] is False
