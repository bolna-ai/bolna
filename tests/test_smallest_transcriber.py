"""Unit tests for SmallestTranscriber provider audio-param matrix.

The constructor calls _configure_audio_params(), which reads
`connected_via_dashboard` for providers without an explicit branch. That
attribute used to be assigned AFTER the call, so any provider falling through
the branch chain (freeswitch webcalls) crashed with AttributeError at
transcriber setup and the whole call died ~3s in. These tests pin the full
provider matrix so a branch regression is loud, not silent.

Constructor-only: no network happens until run().
"""

import pytest

from bolna.transcriber.smallest_transcriber import SmallestTranscriber


def _transcriber(provider, **kwargs):
    return SmallestTranscriber(
        telephony_provider=provider,
        stream=True,
        transcriber_key="test-key",
        **kwargs,
    )


@pytest.mark.parametrize("provider", ["twilio", "sip-trunk"])
def test_mulaw_telephony_providers(provider):
    t = _transcriber(provider)
    assert t.encoding == "mulaw"
    assert t.sampling_rate == 8000
    assert t.audio_frame_duration == 0.2


@pytest.mark.parametrize("provider", ["exotel", "plivo"])
def test_linear16_telephony_providers(provider):
    t = _transcriber(provider)
    assert t.encoding == "linear16"
    assert t.sampling_rate == 8000
    assert t.audio_frame_duration == 0.2


def test_web_based_call():
    t = _transcriber("web_based_call")
    assert t.encoding == "linear16"
    assert t.sampling_rate == 16000
    assert t.audio_frame_duration == 0.256


def test_freeswitch_webcall_does_not_crash_and_uses_16k_linear16():
    # the regression case: provider="freeswitch" had no branch and fell through to
    # `elif not self.connected_via_dashboard` before the attribute existed
    t = _transcriber("freeswitch")
    assert t.encoding == "linear16"
    assert t.sampling_rate == 16000
    assert t.audio_frame_duration == 0.2


def test_playground_batch_mode():
    t = _transcriber("playground")
    assert t.encoding == "linear16"
    assert t.sampling_rate == 8000
    assert t.audio_frame_duration == 0.0


@pytest.mark.parametrize("enforce_streaming", [True, False])
def test_unknown_provider_never_crashes_on_dashboard_flag(enforce_streaming):
    # any provider without an explicit branch must resolve via connected_via_dashboard,
    # which has to exist by the time _configure_audio_params runs
    t = _transcriber("default", enforce_streaming=enforce_streaming)
    assert t.connected_via_dashboard is enforce_streaming
    assert t.encoding == "linear16"
    assert t.sampling_rate == 16000
