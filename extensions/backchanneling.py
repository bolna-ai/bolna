"""
extensions/backchanneling.py -- ambient noise and backchanneling audio presets.

Backchanneling provides subtle environmental audio (office noise, keyboard
typing, breathing sounds, etc.) that is mixed into the TTS output to make
the AI assistant sound more natural during pauses.

Audio presets are stored as small WAV/MP3 files in S3 under a known prefix.
This module loads them on demand and caches the binary content in memory so
repeated calls within the same process don't re-fetch from S3.

Usage::

    audio_bytes = await get_backchannel_audio("office_ambient")
    presets = await list_presets()
"""

from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from aiobotocore.session import AioSession
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# S3 bucket and prefix for backchannel audio files
BUCKET_NAME = os.getenv("BUCKET_NAME", "kallabot-recordings")
_S3_PREFIX = "backchanneling/"

# In-memory cache: preset_name -> bytes
_audio_cache: dict[str, bytes] = {}

# Known presets (name -> S3 key suffix).
# Extend this dict to register new audio presets.
PRESETS: dict[str, dict[str, Any]] = {
    "office_ambient": {
        "key": "office_ambient.wav",
        "description": "Soft office background noise (keyboard, muffled voices)",
        "format": "wav",
    },
    "keyboard_typing": {
        "key": "keyboard_typing.wav",
        "description": "Intermittent keyboard typing sounds",
        "format": "wav",
    },
    "soft_breathing": {
        "key": "soft_breathing.wav",
        "description": "Subtle breathing sounds for natural pauses",
        "format": "wav",
    },
    "cafe_ambience": {
        "key": "cafe_ambience.wav",
        "description": "Light cafe background ambience",
        "format": "wav",
    },
    "paper_rustling": {
        "key": "paper_rustling.wav",
        "description": "Occasional paper rustling sounds",
        "format": "wav",
    },
    "silence": {
        "key": "silence.wav",
        "description": "Comfortable silence (no background noise)",
        "format": "wav",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_backchannel_audio(
    preset_name: str,
    *,
    bucket_name: str | None = None,
    use_cache: bool = True,
) -> bytes | None:
    """Load a backchannel audio preset from S3 (or in-memory cache).

    Parameters
    ----------
    preset_name : str
        One of the keys in :data:`PRESETS` (e.g. ``"office_ambient"``).
    bucket_name : str | None
        Override the default S3 bucket.
    use_cache : bool
        If ``True`` (default), return the cached bytes when available
        instead of re-fetching from S3.

    Returns
    -------
    bytes | None
        Raw audio bytes, or ``None`` if the preset is unknown or the S3
        object could not be fetched.
    """
    if preset_name not in PRESETS:
        logger.warning("Unknown backchannel preset: %s", preset_name)
        return None

    # Check in-memory cache
    if use_cache and preset_name in _audio_cache:
        logger.debug("Backchannel cache hit for %s", preset_name)
        return _audio_cache[preset_name]

    preset_info = PRESETS[preset_name]
    s3_key = f"{_S3_PREFIX}{preset_info['key']}"
    bucket = bucket_name or BUCKET_NAME

    audio_bytes = await _fetch_from_s3(bucket, s3_key)
    if audio_bytes is None:
        return None

    # Populate cache
    if use_cache:
        _audio_cache[preset_name] = audio_bytes
        logger.debug("Cached backchannel audio for %s (%d bytes)", preset_name, len(audio_bytes))

    return audio_bytes


async def list_presets() -> list[dict[str, Any]]:
    """Return metadata for all available backchannel audio presets.

    Returns
    -------
    list[dict]
        Each dict contains ``name``, ``description``, and ``format``.
    """
    return [
        {
            "name": name,
            "description": info["description"],
            "format": info["format"],
        }
        for name, info in PRESETS.items()
    ]


def clear_cache() -> None:
    """Flush the in-memory audio cache.

    Useful for testing or when presets are updated in S3.
    """
    _audio_cache.clear()
    logger.info("Backchannel audio cache cleared")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_from_s3(bucket: str, key: str) -> bytes | None:
    """Fetch raw bytes from an S3 object."""
    session = AioSession()
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(session.create_client("s3"))
        try:
            response = await client.get_object(Bucket=bucket, Key=key)
            body = await response["Body"].read()
            logger.info("Fetched backchannel audio from s3://%s/%s (%d bytes)", bucket, key, len(body))
            return body
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 get_object failed for backchannel %s: %s", key, exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error fetching backchannel %s: %s", key, exc)
            return None
