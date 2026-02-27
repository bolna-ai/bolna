import io
import math
import audioop
import numpy as np
import os

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class AmbientNoiseMixer:
    """
    Overlays ambient background noise onto outgoing TTS audio chunks.

    Pre-loads the background noise file once at init and maintains a playback
    cursor so the noise loops seamlessly across successive mix() calls.

    Supports mulaw, ulaw, pcm (int16), and wav audio formats.
    """

    def __init__(self, noise_pcm_dir=None, snr_db=10, noise_db=-12):
        """
        Args:
            noise_pcm_dir: Directory containing pre-converted .npy noise files
                           (noise_8000.npy, noise_16000.npy, etc.).
                           Defaults to AMBIENT_NOISE_PCM_DIR env var or
                           helpers/noise_pcm/ in the repo.
            snr_db: Signal-to-noise ratio in dB. Higher = quieter background.
                    20dB is subtle, 10dB is prominent.
        """
        self.snr_db = snr_db
        self.noise_db = noise_db
        self.enabled = False
        self._cursor = 0  # Playback position in samples (mono)
        self._loaded = False

        # Pre-loaded background noise as int16 numpy arrays, keyed by sample rate
        self._bg_samples = {}  # {sample_rate: np.ndarray of int16}

        if noise_pcm_dir is None:
            noise_pcm_dir = os.getenv(
                "AMBIENT_NOISE_PCM_DIR",
                os.path.join(os.path.dirname(__file__), "noise_pcm")
            )

        noise_pcm_dir = os.path.abspath(noise_pcm_dir)

        if not os.path.isdir(noise_pcm_dir):
            logger.warning(
                f"Ambient noise PCM directory not found at {noise_pcm_dir}. "
                f"Ambient noise overlay will be disabled."
            )
            return

        try:
            self._load_npy_files(noise_pcm_dir)
            self._loaded = True
            logger.info(
                f"AmbientNoiseMixer initialized from {noise_pcm_dir}, "
                f"SNR={snr_db}dB"
            )
        except Exception as e:
            logger.error(f"Failed to load ambient noise npy files: {e}. Overlay disabled.")

    def _load_npy_files(self, pcm_dir):
        """Load pre-converted .npy noise files keyed by sample rate."""
        for rate in (8000, 16000, 24000, 44100):
            npy_path = os.path.join(pcm_dir, f"noise_{rate}.npy")
            if os.path.exists(npy_path):
                samples = np.load(npy_path)
                self._bg_samples[rate] = samples.astype(np.int16)
                logger.info(
                    f"Loaded ambient noise at {rate}Hz: "
                    f"{len(samples)} samples ({len(samples)/rate:.1f}s)"
                )
            else:
                logger.warning(f"Noise file not found: {npy_path}")

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

    def generate_noise_chunk(self, duration_ms, audio_format, sample_rate=8000):
        """
        Generate a standalone noise-only audio chunk for continuous background playback.

        Uses the same shared cursor as mix() so noise is seamless across
        noise-only frames and agent-overlaid frames.

        Args:
            duration_ms: Duration of the chunk in milliseconds.
            audio_format: One of 'mulaw', 'ulaw', 'pcm'.
            sample_rate: Sample rate (default 8000 for telephony).

        Returns:
            Raw audio bytes of noise-only audio in the specified format.
        """


        if not self.enabled or not self._bg_samples:
            num_samples = int(sample_rate * duration_ms / 1000)
            silence = np.zeros(num_samples, dtype=np.int16)
            if audio_format in ('mulaw', 'ulaw'):
                return audioop.lin2ulaw(silence.tobytes(), 2)
            return silence.tobytes()

        num_samples = int(sample_rate * duration_ms / 1000)
        bg_samples = self._get_bg_slice(num_samples, sample_rate)

        # Fixed gain — no voice signal to reference, so use snr_db directly
        gain = 10 ** (self.noise_db / 20.0)
        adjusted = (bg_samples.astype(np.float64) * gain)
        adjusted = np.clip(adjusted, -32768, 32767).astype(np.int16)

        if audio_format in ('mulaw', 'ulaw'):
            return audioop.lin2ulaw(adjusted.tobytes(), 2)
        return adjusted.tobytes()

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


def main():
    import wave
    noise_mixer = AmbientNoiseMixer()
    noise_mixer.enabled = True
    
    # take a wav file as input, mix it, and store it in the same directory
    
    file = "rec.wav"

    # load the bytes using wave
    with wave.open(file, "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())
        params = wf.getparams()
    
    # mix it
    mixed_audio = noise_mixer._mix_pcm(pcm_data, params.framerate)
    
    # store it
    with wave.open("rec_mixed.wav", "wb") as wf:
        wf.setparams(params)
        wf.writeframes(mixed_audio)

if __name__ == "__main__":
    main()
