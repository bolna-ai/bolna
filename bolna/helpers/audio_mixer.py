"""
AmbientNoiseMixer — continuous, seamless ambient noise mixing for voice calls.

The mixer maintains a single read-pointer into a looping PCM buffer so that
every consumer (content-audio mixing *and* silence-fill packets) sees a
continuous, gap-free noise stream.
"""

import io
import numpy as np
from scipy.io import wavfile
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class AmbientNoiseMixer:
    """Mix a looping ambient-noise track into outgoing audio.

    Parameters
    ----------
    pcm_data : bytes
        Raw **signed 16-bit mono PCM** samples at *sample_rate*.
    volume : float
        Gain applied to the noise before mixing (0.0–1.0).
        0.15 is a reasonable "barely audible background" default.
    sample_rate : int
        The playback sample rate (must match the content audio rate).
    """

    def __init__(self, pcm_data: bytes, volume: float = 0.15, sample_rate: int = 8000):
        if not pcm_data or len(pcm_data) < 2:
            raise ValueError("pcm_data must contain at least one sample")

        self.sample_rate = sample_rate
        self.volume = np.float64(max(0.0, min(1.0, volume)))
        # Store as float64 for mixing precision; the copy avoids repeated casts.
        self._data = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
        self._length = len(self._data)
        self._position = 0

        logger.info(
            f"AmbientNoiseMixer initialized: {self._length} samples "
            f"({self._length / sample_rate:.1f}s), volume={self.volume:.2f}, "
            f"rate={sample_rate}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance(self, num_samples: int) -> np.ndarray:
        """Return *num_samples* of ambient noise (float64) and advance the
        read pointer, looping seamlessly over the buffer boundary."""
        result = np.empty(num_samples, dtype=np.float64)
        remaining = num_samples
        write_pos = 0
        while remaining > 0:
            available = min(remaining, self._length - self._position)
            result[write_pos:write_pos + available] = self._data[self._position:self._position + available]
            self._position = (self._position + available) % self._length
            write_pos += available
            remaining -= available
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mix_into(self, content_pcm: bytes) -> bytes:
        """Mix ambient noise into *content_pcm* (int16 PCM) and return the
        mixed result as int16 PCM bytes.

        The read-pointer advances by ``len(content_pcm) // 2`` samples so that
        any subsequent ``get_chunk`` call continues seamlessly.
        """
        if not content_pcm or len(content_pcm) < 2:
            return content_pcm

        content = np.frombuffer(content_pcm, dtype=np.int16).astype(np.float64)
        noise = self._advance(len(content)) * self.volume
        mixed = np.clip(content + noise, -32768, 32767).astype(np.int16)
        return mixed.tobytes()

    def mix_into_mulaw(self, mulaw_bytes: bytes) -> bytes:
        """Mix ambient noise into *mulaw_bytes* (8-bit mu-law, 1 byte/sample).

        Converts mulaw → int16 PCM, mixes noise, converts back to mulaw.
        """
        import audioop
        if not mulaw_bytes or len(mulaw_bytes) < 1:
            return mulaw_bytes

        # mulaw → int16 PCM (2 bytes/sample)
        pcm_data = audioop.ulaw2lin(mulaw_bytes, 2)
        # Mix noise into PCM
        mixed_pcm = self.mix_into(pcm_data)
        # int16 PCM → mulaw
        return audioop.lin2ulaw(mixed_pcm, 2)

    def get_chunk(self, duration_seconds: float) -> bytes:
        """Return a pure ambient-noise chunk for *duration_seconds* as int16
        PCM bytes.  Used to fill silence when no content audio is being sent.
        """
        num_samples = int(duration_seconds * self.sample_rate)
        if num_samples <= 0:
            return b""
        noise = self._advance(num_samples) * self.volume
        return np.clip(noise, -32768, 32767).astype(np.int16).tobytes()

    def set_volume(self, volume: float) -> None:
        """Hot-update the volume without reloading the audio."""
        self.volume = np.float64(max(0.0, min(1.0, volume)))

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_wav_bytes(cls, wav_bytes: bytes, volume: float = 0.15,
                       target_sample_rate: int = 8000) -> "AmbientNoiseMixer":
        """Create a mixer from raw WAV file bytes.

        The WAV is resampled to *target_sample_rate* if necessary and
        converted to mono int16 PCM.
        """
        import scipy.signal
        import math

        buf = io.BytesIO(wav_bytes)
        rate, data = wavfile.read(buf)

        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Ensure int16
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)

        # Resample if needed
        if rate != target_sample_rate:
            g = math.gcd(rate, target_sample_rate)
            data_f = data.astype(np.float64)
            data_resampled = scipy.signal.resample_poly(
                data_f, target_sample_rate // g, rate // g
            )
            data = np.clip(data_resampled, -32768, 32767).astype(np.int16)
            logger.info(f"Resampled ambient noise from {rate}Hz to {target_sample_rate}Hz")

        return cls(data.tobytes(), volume=volume, sample_rate=target_sample_rate)
