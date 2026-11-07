"""StreamingPcmResampler: audio resampled piece by piece must match resampling it whole,
or every piece boundary becomes an audible click."""

import numpy as np
import pytest

from bolna.helpers.utils import StreamingPcmResampler, resample


def _signal(rate, seconds=1.0, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    x = 6000 * np.sin(2 * np.pi * 300 * t) + 3000 * np.sin(2 * np.pi * 1700 * t) + rng.normal(0, 500, len(t))
    return x.astype(np.int16)


def _stream(resampler, pcm, sizes):
    out, i, k = [], 0, 0
    while i < len(pcm):
        n = sizes[k % len(sizes)]
        out.append(resampler.process(pcm[i : i + n].tobytes()))
        i, k = i + n, k + 1
    return np.frombuffer(b"".join(out), dtype=np.int16)


@pytest.mark.parametrize("source_rate", [24000, 22050, 16000, 44100])
@pytest.mark.parametrize("sizes", [[2048], [1], [7, 4096, 13, 999], [1881, 2048, 300]])
def test_piecewise_output_matches_whole_stream_resampling(source_rate, sizes):
    pcm = _signal(source_rate)
    whole = np.frombuffer(resample(pcm.tobytes(), 8000, format="pcm", original_sample_rate=source_rate), np.int16)
    streamed = _stream(StreamingPcmResampler(source_rate, 8000), pcm, sizes)
    # Everything but a short held-back tail is out, and it is the whole-stream result.
    assert 0 <= len(whole) - len(streamed) <= 12
    assert np.abs(streamed.astype(int) - whole[: len(streamed)].astype(int)).max() <= 1


def test_held_back_samples_come_out_with_the_next_piece():
    resampler = StreamingPcmResampler(24000, 8000)
    pcm = _signal(24000)
    first = resampler.process(pcm[:12000].tobytes())
    second = resampler.process(pcm[12000:].tobytes())
    assert len(first) // 2 < 4000 and (len(first) + len(second)) // 2 > 4000


def test_reset_starts_a_fresh_stream():
    pcm = _signal(24000)
    resampler = StreamingPcmResampler(24000, 8000)
    resampler.process(pcm[:5000].tobytes())
    resampler.reset()
    assert resampler.process(pcm.tobytes()) == StreamingPcmResampler(24000, 8000).process(pcm.tobytes())


def test_same_rate_passes_through():
    pcm = _signal(8000).tobytes()
    assert StreamingPcmResampler(8000, 8000).process(pcm) == pcm
    assert StreamingPcmResampler(24000, 24000).process(pcm) == pcm


def test_odd_byte_count_raises_like_one_shot_resample():
    with pytest.raises(ValueError):
        StreamingPcmResampler(24000, 8000).process(b"\x01")
