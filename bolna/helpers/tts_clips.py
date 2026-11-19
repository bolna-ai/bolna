"""One-shot renders of fixed agent lines in a call's wire format, and the clips a process keeps in memory."""

import asyncio
from collections import OrderedDict

from bolna.constants import WEBCALL_TTS_SAMPLE_RATE
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import audio_to_mulaw8k, audio_to_pcm, get_raw_audio_bytes, store_raw_audio_bytes

logger = configure_logger(__name__)

# Stored clip bytes a worker process keeps across calls, least recently played evicted first.
CLIP_MEMORY_BYTES = 32 * 1024 * 1024
# Spoken length per character outside which a render is a bad take, not a clip worth storing.
MIN_SECONDS_PER_CHAR = 0.03
MAX_SECONDS_PER_CHAR = 0.2
# Wire audio per message when a clip is played, so no transport gets one oversized frame.
CLIP_PIECE_SECONDS = 1.0
RENDER_TIMEOUT_S = 30

_clips: OrderedDict[str, bytes] = OrderedDict()
_clip_bytes = 0
_loading: set[str] = set()
_background: set[asyncio.Task] = set()


def recall_clip(key):
    """Clip bytes this process already holds for key, or None."""
    clip = _clips.get(key)
    if clip is not None:
        _clips.move_to_end(key)
    return clip


def remember_clip(key, clip):
    global _clip_bytes
    if key in _clips:
        _clips.move_to_end(key)
        return
    _clips[key] = clip
    _clip_bytes += len(clip)
    while _clip_bytes > CLIP_MEMORY_BYTES and len(_clips) > 1:
        _, evicted = _clips.popitem(last=False)
        _clip_bytes -= len(evicted)


def wire_seconds(audio, mulaw_wire):
    return len(audio) / 8000 if mulaw_wire else len(audio) / (2 * WEBCALL_TTS_SAMPLE_RATE)


def plausible_length(seconds, text):
    return MIN_SECONDS_PER_CHAR * len(text) <= seconds <= MAX_SECONDS_PER_CHAR * len(text) + 1.0


def clip_pieces(clip, mulaw_wire):
    """The clip cut into transport-sized pieces on sample boundaries."""
    size = int(CLIP_PIECE_SECONDS * (8000 if mulaw_wire else 2 * WEBCALL_TTS_SAMPLE_RATE))
    return [clip[i : i + size] for i in range(0, len(clip), size)]


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
    """Render text as one wire-format clip, or None when nothing usable comes back."""
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
    return clip


def load_fixed_line_clip(synth, text, key, mulaw_wire, storage):
    """Bring a fixed line's clip into this process in the background: read it from the store, or render and store it."""
    if key in _loading:
        return
    _loading.add(key)
    task = asyncio.create_task(_load_or_render(synth, text, key, mulaw_wire, storage))
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _load_or_render(synth, text, key, mulaw_wire, storage):
    try:
        clip = await get_raw_audio_bytes(key, **storage)
        if not clip:
            clip = await asyncio.wait_for(render_clip(synth, text, mulaw_wire), RENDER_TIMEOUT_S)
            if not clip:
                return
            seconds = wire_seconds(clip, mulaw_wire)
            if not plausible_length(seconds, text):
                logger.error(f"Fixed-line clip {key} is {seconds:.2f}s for {len(text)} chars, not storing it")
                return
            await store_raw_audio_bytes(clip, key, **storage)
            logger.info(f"Stored fixed-line clip {key} ({len(clip)} bytes)")
        remember_clip(key, clip)
    except Exception as e:
        logger.error(f"Could not load or render fixed-line clip {key}: {e}")
    finally:
        _loading.discard(key)
