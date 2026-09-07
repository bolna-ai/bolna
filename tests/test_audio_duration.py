"""Converting a buffer size to seconds, which is what timing decisions rest on.

Telephony mulaw carries one byte per sample where PCM carries two, so getting the encoding wrong
misjudges when the agent has finished speaking.
"""

from bolna.helpers.utils import calculate_audio_duration


def test_pcm_duration_counts_two_bytes_per_sample():
    assert calculate_audio_duration(32000, 16000) == 1.0


def test_mulaw_duration_counts_one_byte_per_sample():
    """Halving this would have the agent think it finished speaking twice as fast."""
    assert calculate_audio_duration(8000, 8000, format="mulaw") == 1.0


def test_both_mulaw_spellings_agree():
    assert calculate_audio_duration(8000, 8000, format="ulaw") == calculate_audio_duration(8000, 8000, format="mulaw")


def test_channels_and_bit_depth_are_honoured():
    assert calculate_audio_duration(32000, 8000, channels=2) == 1.0
    assert calculate_audio_duration(8000, 8000, bit_depth=8) == 1.0
