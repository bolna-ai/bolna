"""One-shot renders of fixed agent lines in a call's wire format."""

import asyncio

from bolna.constants import WEBCALL_TTS_SAMPLE_RATE
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import audio_to_mulaw8k, audio_to_pcm

logger = configure_logger(__name__)


def convert_clip(synth, audio, mulaw_wire):
    """Decode a one-shot render into the wire format. Blocking, so run it off the event loop."""
    kwargs = {
        "rate_hint": getattr(synth, "sampling_rate", 0) or getattr(synth, "sample_rate", 0) or 8000,
        "format_hint": getattr(synth, "format", "") or "",
    }
    if mulaw_wire:
        clip = audio_to_mulaw8k(audio, **kwargs)
    else:
        clip = audio_to_pcm(audio, target_sample_rate=WEBCALL_TTS_SAMPLE_RATE, **kwargs)
    if clip is None:
        logger.error(
            f"{synth.__class__.__name__} clip is a compressed container pydub can't decode into "
            f"{'mulaw' if mulaw_wire else 'pcm'}, skipping"
        )
    return clip


async def render_clip(synth, text, mulaw_wire):
    """Render text as one wire-format clip, billing its characters, or None when nothing usable comes back."""
    # Under ~50ms is a failed one-shot returning a sentinel, not a clip.
    min_clip_bytes = 400 if mulaw_wire else 2400
    clip = None
    # Prefer a native one-shot in the wire format when the provider offers it.
    if mulaw_wire:
        telephony_one_shot = getattr(synth, "synthesize_telephony_clip", None)
        if telephony_one_shot is not None:
            clip = await telephony_one_shot(text)
    else:
        pcm_one_shot = getattr(synth, "synthesize_pcm_clip", None)
        if pcm_one_shot is not None:
            clip = await pcm_one_shot(text, WEBCALL_TTS_SAMPLE_RATE)
    # Before the fallback, so synthesize() still gets its turn.
    if clip and len(clip) < min_clip_bytes:
        logger.warning(f"{synth.__class__.__name__} one-shot returned {len(clip)}B, falling back to synthesize()")
        clip = None
    if not clip:
        audio = await synth.synthesize(text)
        if not audio:
            return None
        # pydub decode shells out to ffprobe/ffmpeg, so keep it off the event loop.
        clip = await asyncio.to_thread(convert_clip, synth, audio, mulaw_wire)
    if clip and len(clip) < min_clip_bytes:
        logger.error(f"{synth.__class__.__name__} clip is {len(clip)}B, discarding")
        return None
    if clip:
        synth.synthesized_characters = getattr(synth, "synthesized_characters", 0) + len(text)
    return clip
