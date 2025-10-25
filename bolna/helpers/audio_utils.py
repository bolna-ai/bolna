import io
import numpy as np
import audioop
from scipy import signal
from bolna.helpers.logger_config import configure_logger
from bolna.constants import PCM16_SAMPLE_WIDTH, TELEPHONY_SAMPLE_RATE, OPENAI_REALTIME_SAMPLE_RATE

logger = configure_logger(__name__)


def resample_audio(audio_bytes: bytes, from_rate: int, to_rate: int, from_format: str = 'pcm', to_format: str = 'pcm') -> bytes:
    """
    Resample audio between different sample rates and formats.

    Args:
        audio_bytes: Input audio data
        from_rate: Source sample rate (e.g., 8000, 16000, 24000)
        to_rate: Target sample rate
        from_format: Source format ('pcm', 'mulaw')
        to_format: Target format ('pcm', 'mulaw')

    Returns:
        Resampled audio bytes
    """

    # Convert mulaw to PCM if needed
    if from_format == 'mulaw':
        audio_bytes = audioop.ulaw2lin(audio_bytes, PCM16_SAMPLE_WIDTH)

    # Resample if rates differ
    if from_rate != to_rate:
        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Calculate number of output samples
        num_output_samples = int(len(audio_array) * to_rate / from_rate)

        # Resample using scipy
        resampled_array = signal.resample(audio_array, num_output_samples)

        # Convert back to bytes
        audio_bytes = resampled_array.astype(np.int16).tobytes()

    # Convert PCM to mulaw if needed
    if to_format == 'mulaw':
        audio_bytes = audioop.lin2ulaw(audio_bytes, PCM16_SAMPLE_WIDTH)

    return audio_bytes


def convert_mulaw_to_pcm16_24khz(mulaw_8khz_bytes: bytes) -> bytes:
    """
    Convert Twilio's mulaw 8kHz to PCM16 24kHz for OpenAI Realtime.

    Args:
        mulaw_8khz_bytes: mulaw encoded audio at 8kHz

    Returns:
        PCM16 audio at 24kHz
    """
    # Step 1: mulaw to PCM16 8kHz
    pcm_8khz = audioop.ulaw2lin(mulaw_8khz_bytes, PCM16_SAMPLE_WIDTH)

    # Step 2: Resample 8kHz → 24kHz
    pcm_24khz = resample_audio(pcm_8khz, TELEPHONY_SAMPLE_RATE, OPENAI_REALTIME_SAMPLE_RATE, 'pcm', 'pcm')

    return pcm_24khz


def convert_pcm16_24khz_to_mulaw(pcm_24khz_bytes: bytes) -> bytes:
    """
    Convert OpenAI Realtime's PCM16 24kHz to mulaw 8kHz for Twilio.

    Args:
        pcm_24khz_bytes: PCM16 audio at 24kHz

    Returns:
        mulaw encoded audio at 8kHz
    """
    # Step 1: Resample 24kHz → 8kHz
    pcm_8khz = resample_audio(pcm_24khz_bytes, OPENAI_REALTIME_SAMPLE_RATE, TELEPHONY_SAMPLE_RATE, 'pcm', 'pcm')

    # Step 2: PCM16 to mulaw
    mulaw_8khz = audioop.lin2ulaw(pcm_8khz, PCM16_SAMPLE_WIDTH)

    return mulaw_8khz


def convert_pcm16_to_target_format(pcm_bytes: bytes, from_rate: int, target_provider: str) -> tuple:
    """
    Convert PCM16 audio to target telephony provider format.

    Args:
        pcm_bytes: PCM16 audio data
        from_rate: Source sample rate
        target_provider: Target provider ('twilio', 'plivo', 'exotel', 'default')

    Returns:
        Tuple of (converted_bytes, format_string)
    """
    logger.info(f"[AUDIO CONVERSION] {len(pcm_bytes)}B @ {from_rate}Hz → {target_provider}")

    if target_provider == 'twilio':
        # Twilio uses mulaw 8kHz
        pcm_8khz = resample_audio(pcm_bytes, from_rate, TELEPHONY_SAMPLE_RATE, 'pcm', 'pcm') if from_rate != TELEPHONY_SAMPLE_RATE else pcm_bytes
        mulaw = audioop.lin2ulaw(pcm_8khz, PCM16_SAMPLE_WIDTH)
        logger.info(f"[AUDIO CONVERSION] Complete: {len(mulaw)}B mulaw")
        return mulaw, 'mulaw'

    elif target_provider in ['plivo', 'exotel']:
        # Plivo/Exotel use PCM 8kHz
        pcm_8khz = resample_audio(pcm_bytes, from_rate, TELEPHONY_SAMPLE_RATE, 'pcm', 'pcm') if from_rate != TELEPHONY_SAMPLE_RATE else pcm_bytes
        logger.info(f"[AUDIO CONVERSION] Complete: {len(pcm_8khz)}B pcm")
        return pcm_8khz, 'pcm'

    else:
        # Default: keep as-is
        logger.info(f"[AUDIO CONVERSION] No conversion needed")
        return pcm_bytes, 'pcm'

