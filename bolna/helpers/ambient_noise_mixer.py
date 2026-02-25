import io
import math
import audioop
import numpy as np
import os

from pydub import AudioSegment
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class AmbientNoiseMixer:
    """
    Overlays ambient background noise onto outgoing TTS audio chunks.

    Pre-loads the background noise file once at init and maintains a playback
    cursor so the noise loops seamlessly across successive mix() calls.

    Supports mulaw, ulaw, pcm (int16), and wav audio formats.
    """

    def __init__(self, noise_file_path=None, snr_db=20):
        """
        Args:
            noise_file_path: Path to an MP3/WAV background noise file.
                             Defaults to AMBIENT_NOISE_FILE env var or cafe.mp3 in repo root.
            snr_db: Signal-to-noise ratio in dB. Higher = quieter background.
                    20dB is subtle, 10dB is prominent.
        """
        self.snr_db = snr_db
        self.enabled = False
        self._cursor = 0  # Playback position in samples (mono)

        # Pre-loaded background noise as int16 numpy arrays, keyed by sample rate
        self._bg_samples = {}  # {sample_rate: np.ndarray of int16}

        if noise_file_path is None:
            noise_file_path = os.getenv(
                "AMBIENT_NOISE_FILE",
                os.path.join(os.path.dirname(__file__), "..", "..", "cafe.mp3")
            )

        noise_file_path = os.path.abspath(noise_file_path)

        if not os.path.exists(noise_file_path):
            logger.warning(
                f"Ambient noise file not found at {noise_file_path}. "
                f"Ambient noise overlay will be disabled."
            )
            return

        try:
            self._load_noise(noise_file_path)
            self.enabled = True
            logger.info(
                f"AmbientNoiseMixer initialized with {noise_file_path}, "
                f"SNR={snr_db}dB"
            )
        except Exception as e:
            logger.error(f"Failed to load ambient noise file: {e}. Overlay disabled.")

    def _load_noise(self, file_path):
        """Load the noise file and pre-compute PCM int16 arrays at common sample rates."""
        # Load with pydub (supports mp3, wav, etc.)
        audio = AudioSegment.from_file(file_path)
        # Convert to mono
        audio = audio.set_channels(1)

        # Pre-compute at common sample rates used in the system
        for rate in (8000, 16000, 24000, 44100):
            resampled = audio.set_frame_rate(rate).set_sample_width(2)  # 16-bit
            samples = np.frombuffer(resampled.raw_data, dtype=np.int16).copy()
            self._bg_samples[rate] = samples
            logger.info(
                f"Pre-loaded ambient noise at {rate}Hz: "
                f"{len(samples)} samples ({len(samples)/rate:.1f}s)"
            )

    def _get_bg_slice(self, num_samples, sample_rate):
        """
        Get a slice of background noise samples, advancing the cursor.
        Loops seamlessly when reaching the end of the noise file.
        """
        if sample_rate not in self._bg_samples:
            # Find the closest available rate and use it
            available = list(self._bg_samples.keys())
            closest = min(available, key=lambda r: abs(r - sample_rate))
            logger.warning(
                f"No pre-loaded noise at {sample_rate}Hz, using {closest}Hz"
            )
            sample_rate = closest

        bg = self._bg_samples[sample_rate]
        bg_len = len(bg)

        if bg_len == 0:
            return np.zeros(num_samples, dtype=np.int16)

        # Wrap cursor
        self._cursor = self._cursor % bg_len

        if self._cursor + num_samples <= bg_len:
            # Simple slice, no wrap needed
            result = bg[self._cursor: self._cursor + num_samples]
            self._cursor += num_samples
        else:
            # Need to loop around
            remaining = bg_len - self._cursor
            loops_needed = math.ceil((num_samples - remaining) / bg_len)
            extended = np.concatenate(
                [bg[self._cursor:]] + [bg] * loops_needed
            )
            result = extended[:num_samples]
            self._cursor = (self._cursor + num_samples) % bg_len

        return result.copy()

    def _apply_snr(self, voice_samples, bg_samples):
        """
        Adjust background noise level to achieve the target SNR relative to voice.

        Both inputs are int16 numpy arrays. Returns adjusted bg as int16.
        """
        # Compute RMS of voice signal
        voice_float = voice_samples.astype(np.float64)
        voice_rms = np.sqrt(np.mean(voice_float ** 2)) if len(voice_float) > 0 else 0

        if voice_rms == 0:
            # Silent voice chunk — return very quiet background
            return (bg_samples * 0.01).astype(np.int16)

        bg_float = bg_samples.astype(np.float64)
        bg_rms = np.sqrt(np.mean(bg_float ** 2)) if len(bg_float) > 0 else 0

        if bg_rms == 0:
            return bg_samples

        # Desired background RMS based on SNR
        desired_bg_rms = voice_rms / (10 ** (self.snr_db / 20.0))
        gain = desired_bg_rms / bg_rms

        adjusted = bg_float * gain
        # Clip to int16 range
        adjusted = np.clip(adjusted, -32768, 32767).astype(np.int16)
        return adjusted

    def mix(self, audio_bytes, audio_format, sample_rate=8000):
        """
        Overlay ambient noise onto an audio chunk.

        Args:
            audio_bytes: Raw audio bytes in the specified format.
            audio_format: One of 'mulaw', 'ulaw', 'pcm', 'wav'.
            sample_rate: Sample rate of the audio (used for pcm/mulaw/ulaw).

        Returns:
            Mixed audio bytes in the same format as input.
        """
        if not self.enabled:
            return audio_bytes

        # Skip sentinel/marker bytes
        if audio_bytes in (b'\x00', b'\x00\x00', b''):
            return audio_bytes

        try:
            if audio_format in ('mulaw', 'ulaw'):
                return self._mix_mulaw(audio_bytes, sample_rate)
            elif audio_format == 'pcm':
                return self._mix_pcm(audio_bytes, sample_rate)
            elif audio_format == 'wav':
                return self._mix_wav(audio_bytes, sample_rate)
            else:
                logger.warning(f"Unsupported audio format for mixing: {audio_format}")
                return audio_bytes
        except Exception as e:
            logger.error(f"Error mixing ambient noise (format={audio_format}): {e}")
            return audio_bytes

    def _mix_mulaw(self, audio_bytes, sample_rate):
        """Mix noise into mulaw-encoded audio."""
        # Decode mulaw to linear PCM int16
        pcm_bytes = audioop.ulaw2lin(audio_bytes, 2)  # 2 = 16-bit sample width
        voice_samples = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Get background slice
        bg_samples = self._get_bg_slice(len(voice_samples), sample_rate)
        bg_adjusted = self._apply_snr(voice_samples, bg_samples)

        # Mix: add and clip
        mixed = voice_samples.astype(np.int32) + bg_adjusted.astype(np.int32)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        # Re-encode to mulaw
        mixed_bytes = mixed.tobytes()
        return audioop.lin2ulaw(mixed_bytes, 2)

    def _mix_pcm(self, audio_bytes, sample_rate):
        """Mix noise into raw PCM int16 audio."""
        voice_samples = np.frombuffer(audio_bytes, dtype=np.int16)

        bg_samples = self._get_bg_slice(len(voice_samples), sample_rate)
        bg_adjusted = self._apply_snr(voice_samples, bg_samples)

        mixed = voice_samples.astype(np.int32) + bg_adjusted.astype(np.int32)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        return mixed.tobytes()

    def _mix_wav(self, audio_bytes, sample_rate):
        """Mix noise into WAV-formatted audio."""
        import wave

        # Parse WAV header to get parameters
        wav_buf = io.BytesIO(audio_bytes)
        try:
            with wave.open(wav_buf, 'rb') as wf:
                params = wf.getparams()
                pcm_data = wf.readframes(params.nframes)
                actual_rate = params.framerate
                n_channels = params.nchannels
                sampwidth = params.sampwidth
        except Exception:
            # If WAV parsing fails, try treating as raw PCM
            logger.warning("WAV parsing failed, attempting raw PCM mix")
            return self._mix_pcm(audio_bytes, sample_rate)

        if sampwidth != 2:
            logger.warning(f"WAV sample width is {sampwidth}, expected 2. Skipping mix.")
            return audio_bytes

        # Convert to mono if stereo for mixing
        if n_channels == 2:
            pcm_data = audioop.tomono(pcm_data, 2, 0.5, 0.5)

        voice_samples = np.frombuffer(pcm_data, dtype=np.int16)

        bg_samples = self._get_bg_slice(len(voice_samples), actual_rate)
        bg_adjusted = self._apply_snr(voice_samples, bg_samples)

        mixed = voice_samples.astype(np.int32) + bg_adjusted.astype(np.int32)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        mixed_pcm = mixed.tobytes()

        # Convert back to stereo if original was stereo
        if n_channels == 2:
            mixed_pcm = audioop.tostereo(mixed_pcm, 2, 1, 1)

        # Re-wrap in WAV
        out_buf = io.BytesIO()
        with wave.open(out_buf, 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(mixed_pcm)

        return out_buf.getvalue()