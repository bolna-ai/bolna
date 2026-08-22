"""welcome_pcm_upsampled: source-rate aware welcome preparation.

The backend sends the welcome PCM's rate in the ws payload (8000 for telephony and
legacy payloads, 24000 for web calls). Web/freeswitch playback runs at 24kHz:
8kHz welcomes are upsampled exactly as before, 24kHz welcomes pass through untouched.
"""

import base64
import struct

from bolna.agent_manager.task_manager import welcome_pcm_upsampled


def pcm_b64(num_samples: int) -> str:
    pcm = b"".join(struct.pack("<h", (2000 if (i // 3) % 2 else -2000)) for i in range(num_samples))
    return base64.b64encode(pcm).decode()


def test_8k_source_upsampled_to_24k():
    """Legacy behavior: 1s of 8kHz PCM becomes ~1s of 24kHz PCM (3x the samples)."""
    welcome = pcm_b64(8000)
    out = welcome_pcm_upsampled(welcome, 24000, 8000)
    assert abs(len(out) - 8000 * 2 * 3) <= 0.02 * 8000 * 2 * 3


def test_default_source_rate_is_8k():
    """Two-arg call (old payloads without the rate field) behaves exactly as before."""
    welcome = pcm_b64(8000)
    assert welcome_pcm_upsampled(welcome, 24000) == welcome_pcm_upsampled(welcome, 24000, 8000)


def test_24k_source_passes_through_untouched():
    """A welcome already at the playback rate must not be resampled — byte-identical."""
    welcome = pcm_b64(24000)
    out = welcome_pcm_upsampled(welcome, 24000, 24000)
    assert out == base64.b64decode(welcome)


def test_8k_to_8k_passes_through_untouched():
    """Telephony-shaped inputs (source == target == 8k) are also a no-op."""
    welcome = pcm_b64(4000)
    out = welcome_pcm_upsampled(welcome, 8000, 8000)
    assert out == base64.b64decode(welcome)
