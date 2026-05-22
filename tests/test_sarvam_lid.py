"""
Standalone test for SarvamLID language detection.

Usage:
    python tests/test_sarvam_lid.py <path_to_wav>

The WAV file should be mono (any sample rate — auto-detected).
Ideally contains Hindi, Tamil, and Telugu speech segments.

Requires SARVAM_API_KEY env var (or .env file in repo root).
"""

import asyncio
import sys
import time
import wave

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from bolna.lid.sarvam import SarvamLID

CHUNK_DURATION_S = 0.2  # 200ms per chunk, matching sarvam_transcriber


def read_wav_chunks(path: str, chunk_duration_s: float):
    """Read a WAV file and yield raw PCM chunks of chunk_duration_s seconds."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()

        print(f"Audio: {sr}Hz, {n_channels}ch, {sampwidth * 8}-bit, {n_frames / sr:.1f}s total")

        if n_channels != 1:
            raise ValueError(f"Expected mono audio, got {n_channels} channels")

        chunk_frames = int(sr * chunk_duration_s)
        while True:
            data = wf.readframes(chunk_frames)
            if not data:
                break
            yield data, sr, sampwidth


async def run(wav_path: str, fast: bool = False):
    detections = []
    start_time = time.monotonic()

    async def on_language(lang: str, conf: float):
        t = time.monotonic() - start_time
        detections.append((t, lang, conf))
        print(f"  [t={t:.1f}s] detected: {lang} (conf={conf:.2f})")

    chunks = list(read_wav_chunks(wav_path, CHUNK_DURATION_S))
    if not chunks:
        print("ERROR: no audio data read from file")
        return

    _, sr, sampwidth = chunks[0]

    lid = SarvamLID(
        on_language=on_language,
        config={
            "sampling_rate": sr,
            "telephony_provider": "",  # linear16, no mulaw conversion
        },
    )

    print("\nConnecting to Sarvam WS...")
    await lid.start()
    print(f"Connected. Feeding {len(chunks)} chunks ({len(chunks) * CHUNK_DURATION_S:.1f}s of audio)...\n")

    for i, (chunk, _, _) in enumerate(chunks):
        lid.feed(chunk)
        # Sleep to simulate real-time streaming (skip with --fast)
        await asyncio.sleep(0.005 if fast else CHUNK_DURATION_S)

    # Give receiver a moment to flush final detections
    await asyncio.sleep(1.0)
    await lid.stop()

    print(f"\n--- Summary: {len(detections)} detection(s) ---")
    seen = {}
    for t, lang, conf in detections:
        seen[lang] = seen.get(lang, 0) + 1
    for lang, count in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count} detection(s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_sarvam_lid.py <path_to_wav> [--fast]")
        sys.exit(1)
    fast_mode = "--fast" in sys.argv
    asyncio.run(run(sys.argv[1], fast=fast_mode))
