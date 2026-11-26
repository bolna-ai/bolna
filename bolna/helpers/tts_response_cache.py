"""Agent replies replayed from stored audio once the same line has played often enough on one voice."""

import asyncio
import json
import os
import struct
import time
from dataclasses import dataclass, field

from bolna.constants import PREPROCESS_DIR
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.tts_clips import plausible_length, recall_clip, remember_clip
from bolna.helpers.utils import BUCKET_NAME, get_md5_hash, get_s3_file, list_s3_objects, store_file

logger = configure_logger(__name__)

CACHE_ROOT = "tts_cache"
CLIP_SUFFIX = ".clip"
# How long a pod trusts its list of stored clips before listing the store again.
MANIFEST_REFRESH_S = 300


@dataclass
class Manifest:
    keys: set = field(default_factory=set)
    loaded_at: float = 0.0
    loading: asyncio.Task | None = None


@dataclass
class Capture:
    """A live reply whose audio is kept so it can be stored if the line turns out to repeat."""

    object_key: str
    prefix: str
    text: str
    chunks: list = field(default_factory=list)
    audio_format: str | None = None
    wire_format: str | None = None


_manifests: dict[str, Manifest] = {}
_loading: set[str] = set()
_background: set[asyncio.Task] = set()


def encode_clip(audio_format, chunks):
    """One blob: a JSON header of chunk lengths and texts, then the chunks' audio back to back."""
    header = json.dumps({"format": audio_format, "chunks": [[len(audio), text] for audio, text in chunks]}).encode()
    return struct.pack(">I", len(header)) + header + b"".join(audio for audio, _ in chunks)


def decode_clip(blob):
    """(audio_format, [(audio, text_synthesized), ...]) from an encode_clip blob."""
    (header_len,) = struct.unpack(">I", blob[:4])
    header = json.loads(blob[4 : 4 + header_len])
    chunks, offset = [], 4 + header_len
    for length, text in header["chunks"]:
        chunks.append((blob[offset : offset + length], text))
        offset += length
    return header["format"], chunks


class TtsResponseCache:
    """Per-call view of the clips stored for one scope (an organisation) and the play counter behind them."""

    def __init__(self, scope, min_plays, max_age_days, record_play, local=False):
        self.scope = scope
        self.min_plays = min_plays
        self.max_age_s = max_age_days * 86400
        self.record_play = record_play
        self.local = local
        self._store_tasks = set()

    def prefix(self, voice_identity, wire_format):
        return f"{CACHE_ROOT}/{self.scope}/{voice_identity}_{wire_format}/"

    @staticmethod
    def object_key(prefix, text):
        return f"{prefix}{get_md5_hash(text)}{CLIP_SUFFIX}"

    def warm(self, prefix):
        """Start listing the stored clips under prefix unless a fresh list is held or one is loading."""
        manifest = _manifests.setdefault(prefix, Manifest())
        if manifest.loading and not manifest.loading.done():
            return
        if time.time() - manifest.loaded_at < MANIFEST_REFRESH_S:
            return
        manifest.loading = asyncio.create_task(self._load_manifest(prefix, manifest))

    def has(self, object_key, prefix):
        manifest = _manifests.get(prefix)
        return manifest is not None and object_key in manifest.keys

    def fetch(self, object_key):
        """A stored reply's (audio_format, chunks) when this process holds it; otherwise load it in the background, None."""
        blob = recall_clip(object_key)
        if blob is None:
            self._load(object_key)
            return None
        try:
            clip = decode_clip(blob)
        except Exception as e:
            logger.error(f"Unreadable reply clip {object_key}, synthesizing live: {e}")
            return None
        return clip if clip[1] else None

    def _load(self, object_key):
        if object_key in _loading:
            return
        _loading.add(object_key)
        task = asyncio.create_task(self._load_clip(object_key))
        _background.add(task)
        task.add_done_callback(_background.discard)

    async def _load_clip(self, object_key):
        try:
            blob = await self._read(object_key)
            if blob and decode_clip(blob)[1]:
                remember_clip(object_key, blob)
        except Exception as e:
            logger.error(f"Could not load reply clip {object_key}: {e}")
        finally:
            _loading.discard(object_key)

    def finish_capture(self, capture, audio_seconds):
        """Count a completed live play of the line and store its audio once it reaches min_plays."""
        if not plausible_length(audio_seconds, capture.text):
            logger.info(f"Reply audio {audio_seconds:.2f}s does not fit {len(capture.text)} chars, not counting it")
            return
        # Providers that align audio to text must have spoken all of it; a dropped socket ends the turn early.
        spoken = "".join("".join(text for _, text in capture.chunks).split())
        if spoken and spoken != "".join(capture.text.split()):
            logger.info(f"Reply audio covers only part of its text, not counting {capture.object_key}")
            return
        task = asyncio.create_task(self._count_and_store(capture))
        self._store_tasks.add(task)
        task.add_done_callback(self._store_tasks.discard)

    async def _count_and_store(self, capture):
        try:
            plays = await self.record_play(capture.object_key)
            if plays < self.min_plays or self.has(capture.object_key, capture.prefix):
                return
            blob = encode_clip(capture.audio_format, capture.chunks)
            stored = await store_file(
                bucket_name=BUCKET_NAME,
                file_key=capture.object_key,
                file_data=blob,
                content_type="clip",
                local=self.local,
            )
            if not stored:
                return
            _manifests.setdefault(capture.prefix, Manifest()).keys.add(capture.object_key)
            remember_clip(capture.object_key, blob)
            logger.info(f"Stored reply clip {capture.object_key} after {plays} plays ({len(blob)} bytes)")
        except Exception as e:
            logger.error(f"Could not count or store reply clip {capture.object_key}: {e}")

    async def _load_manifest(self, prefix, manifest):
        try:
            cutoff = time.time() - self.max_age_s
            manifest.keys = {key for key, modified in await self._list(prefix) if modified >= cutoff}
            logger.info(f"Loaded {len(manifest.keys)} reply clips under {prefix}")
        except Exception as e:
            logger.error(f"Could not list reply clips under {prefix}: {e}")
        finally:
            # A failed listing also waits out the refresh interval rather than retrying on every reply.
            manifest.loaded_at = time.time()

    async def _list(self, prefix):
        if not self.local:
            return await list_s3_objects(BUCKET_NAME, prefix)
        directory = os.path.join(PREPROCESS_DIR, prefix)
        if not os.path.isdir(directory):
            return []
        return [(f"{prefix}{entry.name}", entry.stat().st_mtime) for entry in os.scandir(directory)]

    async def _read(self, object_key):
        if not self.local:
            return await get_s3_file(BUCKET_NAME, object_key)
        path = os.path.join(PREPROCESS_DIR, object_key)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read()
