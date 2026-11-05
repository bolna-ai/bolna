"""Per-label Deepgram endpoints reach the pool; the top-level endpoint is NOT inherited into legs.

The router (dashboard-backend) stamps only the legs a chosen endpoint can actually serve, so an
unstamped leg must keep its default host rather than inherit an endpoint it may not support.
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


def test_per_label_endpoints_are_preserved(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "multilingual": {
            "en": {
                "provider": "deepgram",
                "model": "nova-3",
                "deepgram_endpoint": "gcp",
            },
            "hi": {
                "provider": "deepgram",
                "model": "nova-2",
                "deepgram_endpoint": "modal",
            },
        },
    }
    _run_setup(monkeypatch, cfg)

    assert cfg["multilingual"]["en"]["deepgram_endpoint"] == "gcp"
    assert cfg["multilingual"]["hi"]["deepgram_endpoint"] == "modal"


def test_top_level_endpoint_not_inherited_into_legs(monkeypatch):
    # An unstamped leg must NOT pick up the top-level endpoint: that inheritance would force an
    # unsupported model onto the wrong endpoint.
    cfg = {
        "provider": "deepgram",
        "deepgram_endpoint": "gcp",
        "multilingual": {
            "en": {"provider": "deepgram", "model": "nova-3", "deepgram_endpoint": "gcp"},
            "hi": {"provider": "deepgram", "model": "flux-general-hi"},  # deliberately left on default
        },
    }
    _run_setup(monkeypatch, cfg)

    assert cfg["multilingual"]["en"]["deepgram_endpoint"] == "gcp"
    assert "deepgram_endpoint" not in cfg["multilingual"]["hi"]


def test_no_override_leaves_labels_untouched(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "multilingual": {"en": {"provider": "deepgram", "model": "nova-2"}},
    }
    _run_setup(monkeypatch, cfg)

    assert "deepgram_endpoint" not in cfg["multilingual"]["en"]


def test_top_level_mip_opt_out_inherited_into_legs(monkeypatch):
    cfg = {
        "provider": "deepgram",
        "mip_opt_out": True,
        "multilingual": {
            "en": {"provider": "deepgram", "model": "nova-3"},
            "hi": {"provider": "deepgram", "model": "nova-2", "mip_opt_out": False},
        },
    }
    _run_setup(monkeypatch, cfg)

    assert cfg["multilingual"]["en"]["mip_opt_out"] is True
    assert cfg["multilingual"]["hi"]["mip_opt_out"] is False
