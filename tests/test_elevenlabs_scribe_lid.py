"""
Standalone test for ElevenLabsScribeLID language detection.

Generates a mixed Hindi / Tamil / Telugu WAV via create_test_audio.py,
then streams it in 50ms chunks to ElevenLabs Scribe v2 Realtime WS.

Generated audio: /tmp/test_elevenlabs_lid.wav

Usage:
    python tests/test_elevenlabs_scribe_lid.py [--fast]

Requires ELEVENLABS_API_KEY env var (or .env file in repo root).
Requires: gtts, pydub, ffmpeg  (same deps as create_test_audio.py)
"""

import asyncio
import io
import os
import sys
import time
import wave

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from bolna.lid.elevenlabs_scribe import ElevenLabsScribeLID

CHUNK_DURATION_S = 0.100  # 100ms per chunk — matches Twilio 10×20ms batch size
TEST_AUDIO_PATH = "/tmp/test_elevenlabs_lid.wav"


# ── Audio generation ──────────────────────────────────────────────────────────


def generate_test_audio(path: str):
    """Generate mixed-language speech WAV using create_test_audio.py logic."""
    from gtts import gTTS
    from pydub import AudioSegment

    segments_def = [
        ("hi", "नमस्ते, मेरा नाम राम है। मैं दिल्ली से हूँ। आज मौसम बहुत अच्छा है।"),
        ("ta", "வணக்கம், என் பெயர் குமார். நான் சென்னையில் இருந்து வருகிறேன்."),
        ("te", "నమస్కారం, నా పేరు రాజు. నేను హైదరాబాద్ నుండి వచ్చాను."),
        ("hi", "क्या आप मुझे बता सकते हैं कि यहाँ का सबसे अच्छा रेस्टोरेंट कौन सा है?"),
    ]

    silence = AudioSegment.silent(duration=800, frame_rate=16000)
    combined = AudioSegment.silent(duration=500, frame_rate=16000)

    for i, (lang, text) in enumerate(segments_def):
        print(f"  [{i + 1}/{len(segments_def)}] {lang}: {text[:50]}...")
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        seg = AudioSegment.from_mp3(buf)
        seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        combined += seg + silence

    combined += AudioSegment.silent(duration=500, frame_rate=16000)
    combined.export(path, format="wav")
    print(f"\n[audio] Saved: {path} ({len(combined) / 1000:.1f}s, mono 16kHz 16-bit)\n")


# ── WAV reader ────────────────────────────────────────────────────────────────


def read_wav_chunks(path: str, chunk_duration_s: float):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()

        print(f"[wav] {sr}Hz, {n_channels}ch, {sampwidth * 8}-bit, {n_frames / sr:.1f}s total")

        if n_channels != 1:
            raise ValueError(f"Expected mono audio, got {n_channels} channels")

        chunk_frames = int(sr * chunk_duration_s)
        while True:
            data = wf.readframes(chunk_frames)
            if not data:
                break
            yield data, sr


# ── Main test ─────────────────────────────────────────────────────────────────


async def run(fast: bool = False):
    print("\n=== ElevenLabs Scribe v2 Realtime LID Test ===\n")

    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"[audio] Generating test audio → {TEST_AUDIO_PATH}")
        generate_test_audio(TEST_AUDIO_PATH)
    else:
        print(f"[audio] Using existing file: {TEST_AUDIO_PATH}")
        print("        (delete it to regenerate)\n")

    detections = []
    start_time = time.monotonic()

    async def on_language(lang: str, conf):
        t = time.monotonic() - start_time
        detections.append((t, lang, conf))
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        print(f"  [t={t:.1f}s] detected: {lang} (conf={conf_str})")

    chunks = list(read_wav_chunks(TEST_AUDIO_PATH, CHUNK_DURATION_S))
    if not chunks:
        print("ERROR: no audio data read from file")
        return

    _, sr = chunks[0]

    lid = ElevenLabsScribeLID(
        on_language=on_language,
        config={
            "sampling_rate": sr,
            "telephony_provider": "",
        },
    )

    print("Connecting to ElevenLabs Scribe v2 Realtime WS...")
    await lid.start()
    print(f"Connected. Streaming {len(chunks)} chunks ({len(chunks) * CHUNK_DURATION_S:.1f}s)...\n")

    for chunk, _ in chunks:
        lid.feed(chunk)
        await asyncio.sleep(0.010 if fast else CHUNK_DURATION_S)

    # Wait for final committed transcripts to flush
    await asyncio.sleep(5.0)
    await lid.stop()

    print(f"\n--- Summary: {len(detections)} detection(s) ---")
    seen = {}
    for t, lang, conf in detections:
        seen[lang] = seen.get(lang, 0) + 1
    for lang, count in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count} detection(s)")


if __name__ == "__main__":
    fast_mode = "--fast" in sys.argv
    asyncio.run(run(fast=fast_mode))
