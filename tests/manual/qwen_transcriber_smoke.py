"""
Standalone smoke-test for QwenTranscriber (Qwen3-ASR realtime, Alibaba Model Studio).

Needs a live credential, so it lives in tests/manual/ and pytest does not collect it.
The unit suite (tests/test_qwen_transcriber.py) covers everything that can be checked
offline; this script covers the three things only the real endpoint can confirm:

  1. the handshake      — URL, Authorization header, and session.update shape are accepted
  2. the audio contract — Qwen actually decodes what we put on the wire (pcm, right rate)
  3. the turn lifecycle — server VAD really does produce speech_started → text → completed

Run (cheapest first):

    export QWEN_API_KEY=sk-...          # or DASHSCOPE_API_KEY

    # 1. handshake only — no audio, ~2s. Isolates auth/URL/config errors.
    python tests/manual/qwen_transcriber_smoke.py --handshake

    # 2. full turn over the default 16kHz linear16 path (web/default provider)
    python tests/manual/qwen_transcriber_smoke.py

    # 3. same audio over the 8kHz mu-law telephony path (Twilio/sip-trunk shape).
    #    This is the one that proves the ulaw2lin decode is right — Qwen rejects mu-law,
    #    so if the conversion were wrong this transcribes noise rather than erroring.
    python tests/manual/qwen_transcriber_smoke.py --telephony

    # 4. the OPEN-WEIGHTS path — no Alibaba account needed. Any OpenAI-compatible host.
    #    OPENROUTER_API_KEY / DEEPINFRA_API_KEY in the environment are picked up automatically.
    python tests/manual/qwen_transcriber_smoke.py --batch
    #    DeepInfra ($5 free on signup, no card) — the upstream host behind OpenRouter's listing:
    python tests/manual/qwen_transcriber_smoke.py --batch \\
        --base-url https://api.deepinfra.com/v1/openai --model Qwen/Qwen3-ASR-1.7B
    python tests/manual/qwen_transcriber_smoke.py --batch \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-ASR-1.7B   # self-hosted vLLM

Options:
    --handshake        connect, assert session.created/updated, exit (realtime only)
    --telephony        provider=twilio → 8kHz mu-law input (default: 16kHz linear16)
    --batch            open weights over /v1/audio/transcriptions instead of the realtime ws
    --base-url URL     OpenAI-compatible base for --batch (default: OpenRouter)
    --language xx      pin a language (default: en; use "auto" for auto-detect)
    --model NAME       override the model id
    <path.wav>         any mono/stereo PCM WAV, any sample rate (resampled here)

Key lookup: --batch prefers QWEN_ASR_BATCH_KEY (the host's own key) and falls back to
QWEN_API_KEY / DASHSCOPE_API_KEY; the realtime path uses the latter two.

Default audio is tests/manual/test_speech.wav (4.5s of English, 24kHz mono).
"""

import argparse
import asyncio
import audioop
import json
import os
import pathlib
import struct
import sys
import time
import wave

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from bolna.constants import QWEN_ASR_DEFAULT_MODEL  # noqa: E402
from bolna.transcriber.qwen_transcriber import QwenTranscriber  # noqa: E402

# ── tunables ──────────────────────────────────────────────────────────────────
CHUNK_MS = 20  # audio feed granularity, matching a telephony frame
SILENCE_AFTER_MS = 1600  # silence fed after speech so Qwen's server VAD closes the turn
MAX_WAV_SECONDS = 15  # cap playback
TIMEOUT_S = 60  # max total wait for events
HANDSHAKE_WAIT_S = 5.0  # how long to listen for session.created/updated
# ─────────────────────────────────────────────────────────────────────────────


def _load_wav_mono(path):
    """Return (pcm16_mono_bytes, sample_rate) for a WAV file.

    Reads to EOF rather than trusting getnframes() — test_speech.wav carries a
    streaming header whose frame count is 0x7FFFFFFF.
    """
    with wave.open(path, "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        if width != 2:
            print(f"[error] {path}: {width * 8}-bit WAV; this script expects 16-bit PCM")
            sys.exit(1)
        raw = wf.readframes(int(rate * MAX_WAV_SECONDS))
    if channels == 2:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    dur = len(raw) / 2 / rate
    print(f"[wav] {path}: {rate}Hz {channels}ch 16bit — {dur:.2f}s of audio")
    return raw, rate


def _prepare_audio(pcm, in_rate, target_rate, to_mulaw):
    """Resample to the rate the transcriber is configured for, and optionally mu-law encode.

    The transcriber deliberately does NOT resample (Qwen takes 8k and 16k natively), so
    getting this right here is the harness's job — exactly as an input handler would.
    """
    if in_rate != target_rate:
        pcm, _ = audioop.ratecv(pcm, 2, 1, in_rate, target_rate, None)
        print(f"[prep] resampled {in_rate}Hz → {target_rate}Hz")
    if to_mulaw:
        pcm = audioop.lin2ulaw(pcm, 2)
        print("[prep] encoded linear16 → mu-law (as Twilio would deliver it)")
    return pcm


def _chunks(payload, bytes_per_chunk):
    for i in range(0, len(payload), bytes_per_chunk):
        yield payload[i : i + bytes_per_chunk]


async def feed_audio(audio_queue, payload, bytes_per_chunk, silence_byte):
    """Push speech, then silence to trigger server VAD, then EOS — paced in real time."""
    meta = {"request_id": "qwen-smoke-001", "call_sid": "qwen-smoke-call"}

    speech = list(_chunks(payload, bytes_per_chunk))
    print(f"[feed] {len(speech)} speech chunks of {CHUNK_MS}ms")
    for chunk in speech:
        await audio_queue.put({"data": chunk, "meta_info": meta})
        await asyncio.sleep(CHUNK_MS / 1000)

    n_silence = SILENCE_AFTER_MS // CHUNK_MS
    print(f"[feed] {n_silence} silence chunks ({SILENCE_AFTER_MS}ms) so server VAD closes the turn")
    silence = silence_byte * bytes_per_chunk
    for _ in range(n_silence):
        await audio_queue.put({"data": silence, "meta_info": meta})
        await asyncio.sleep(CHUNK_MS / 1000)

    print("[feed] sending EOS")
    await audio_queue.put({"data": b"", "meta_info": {**meta, "eos": True}})


async def collect_events(output_queue, timeout):
    """Drain the transcriber output queue until connection_closed or timeout."""
    events, transcripts = [], []
    deadline = time.time() + timeout
    print()
    print("─" * 64)
    print("TRANSCRIBER EVENTS")
    print("─" * 64)
    while time.time() < deadline:
        try:
            packet = await asyncio.wait_for(output_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        data, meta = packet.get("data"), packet.get("meta_info", {})

        if data == "speech_started":
            print("[event] SPEECH STARTED")
            events.append("speech_started")

        elif isinstance(data, dict) and data.get("type") == "interim_transcript_received":
            print(f"[interim] {data.get('content', '')}", end="\r")
            events.append("interim")

        elif isinstance(data, dict) and data.get("type") == "transcript":
            text = data.get("content", "")
            transcripts.append(text)
            print()
            print(f"[event] TRANSCRIPT: {text}")
            for label, key in (
                ("first-result latency ", "transcriber_first_result_latency"),
                ("user_stop_offset_ms  ", "user_stop_offset_ms"),
                ("detected language    ", "transcriber_detected_language"),
                ("detected emotion     ", "transcriber_detected_emotion"),
            ):
                if meta.get(key) is not None:
                    print(f"        {label}: {meta[key]}")
            if data.get("force_finalized"):
                print("        NOTE: force-finalized by the watchdog (no .completed arrived)")
            events.append("force_finalized" if data.get("force_finalized") else "transcript")

        elif isinstance(data, dict) and data.get("type") == "speech_ended":
            print("\n[event] SPEECH ENDED (turn closed with no text)")
            events.append("speech_ended")

        elif data == "transcriber_connection_closed":
            err = meta.get("connection_error")
            print()
            print(f"[event] CONNECTION CLOSED ({'error: ' + str(err) if err else 'clean'})")
            events.append("connection_closed")
            break

        else:
            print(f"[event] OTHER: {data!r}")

    print("─" * 64)
    if transcripts:
        print()
        print("FULL TRANSCRIPT")
        print("─" * 64)
        for i, t in enumerate(transcripts, 1):
            print(f"  Turn {i}: {t}")
        print("─" * 64)
    return events


def _assert(condition, message):
    print(f"{'[pass]' if condition else '[FAIL]'} {message}")
    return bool(condition)


def _build(args, api_key, provider, sampling_rate):
    kw = {}
    if args.batch:
        # leave `model` unset so the realtime default is swapped for a servable batch id
        kw["base_url"] = args.base_url
        if args.model != QWEN_ASR_DEFAULT_MODEL:
            kw["model"] = args.model
    else:
        kw["model"] = args.model
    return QwenTranscriber(
        telephony_provider=provider,
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        language=args.language,
        sampling_rate=sampling_rate,
        endpointing=600,
        stream=not args.batch,
        transcriber_key=api_key,
        **kw,
    )


async def run_handshake(args, api_key):
    """Connect, send session.update, and prove the server accepted the session."""
    t = _build(args, api_key, "default", 16000)
    print(f"[test] handshake against {t.get_qwen_ws_url()}")
    print(f"[test] session.update = {json.dumps(t._build_session_config()['session'])}")

    try:
        ws = await t.qwen_connect()
    except Exception as e:
        print(f"\n[FAIL] could not connect: {e}")
        print("       401/403 → bad or wrong-region key. Check QWEN_ASR_HOST (intl vs Beijing).")
        return 1

    print("\n[test] connected; listening for server events…")
    seen = []
    deadline = time.time() + HANDSHAKE_WAIT_S
    try:
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
            except asyncio.TimeoutError:
                break
            evt = json.loads(raw)
            seen.append(evt.get("type"))
            print(f"[event] {evt.get('type')}: {json.dumps(evt)[:300]}")
            if evt.get("type") == "error":
                print("\n[FAIL] server rejected the session — the error above names the bad field.")
                return 1
            if evt.get("type") == "session.updated":
                break
    finally:
        await ws.close()

    print()
    ok = _assert(any(s in seen for s in ("session.created", "session.updated")), "server acknowledged the session")
    if ok:
        print("\nHandshake OK — URL, auth header and session config are all accepted.")
        print("Next: run without --handshake to push real audio through.")
    return 0 if ok else 1


async def run_full(args, api_key):
    """Feed a real WAV through the real transcriber and check the turn lifecycle."""
    provider = "twilio" if args.telephony else "default"
    rate = 8000 if args.telephony else 16000

    t = _build(args, api_key, provider, rate)
    print(
        f"[test] provider={provider} encoding={t.encoding} sample_rate={t.sampling_rate} "
        f"stream={t.stream} model={t.model}"
    )
    if t.stream:
        print(f"[test] realtime ws={t.get_qwen_ws_url()} silence_duration_ms={t.silence_duration_ms}")
    else:
        print(
            f"[test] batch http={t.transcriptions_url} local endpointing={t.endpointing_ms}ms "
            f"rms_threshold={t.speech_rms_threshold}"
        )

    pcm, in_rate = _load_wav_mono(args.wav)
    payload = _prepare_audio(pcm, in_rate, t.sampling_rate, to_mulaw=(t.encoding == "mulaw"))
    # mu-law is 1 byte/sample, linear16 is 2 — and mu-law silence is 0xff, not 0x00.
    width = 1 if t.encoding == "mulaw" else 2
    bytes_per_chunk = int(t.sampling_rate * CHUNK_MS / 1000) * width
    silence_byte = b"\xff" if t.encoding == "mulaw" else b"\x00"

    await t.run()
    feeder = asyncio.create_task(feed_audio(t.input_queue, payload, bytes_per_chunk, silence_byte))
    events = await collect_events(t.transcriber_output_queue, TIMEOUT_S)
    await feeder
    await t.cleanup()

    print()
    print("ASSERTIONS")
    print("─" * 64)
    ok = True
    ok &= _assert("connection_closed" in events, "connection closed (socket lifecycle completed)")
    vad = "Qwen's server VAD" if t.stream else "the local RMS endpointer"
    ok &= _assert("speech_started" in events, f"speech_started — {vad} heard the audio")
    if t.stream:
        ok &= _assert("interim" in events, "at least one interim (text/stash partials arriving)")
    else:
        print("[skip] interim check — the batch endpoint returns no partials, by design")
    ok &= _assert(
        "transcript" in events or "force_finalized" in events,
        "a final transcript was delivered",
    )
    if "force_finalized" in events and "transcript" not in events:
        print("       ↑ delivered by the WATCHDOG, not a .completed event. The turn still")
        print("         reached the LLM, but check whether QWEN_ASR_COMPLETION_TIMEOUT_S")
        print("         is too tight for this region's latency.")

    if not ok:
        print()
        print("Not all checks passed. Read the events above — the usual causes:")
        print("  · no speech_started  → audio format/rate mismatch; Qwen heard noise, not speech")
        print("  · no interim         → connected but nothing decoded; check sample_rate")
        print("  · connection error   → auth, region, or a rejected session field")
        return 1

    print()
    print("All assertions passed — the live path works end to end.")
    return 0


def main():
    default_wav = str(pathlib.Path(__file__).parent / "test_speech.wav")
    p = argparse.ArgumentParser(description="Live smoke-test for QwenTranscriber.")
    p.add_argument("wav", nargs="?", default=default_wav, help="WAV file to speak (16-bit PCM)")
    p.add_argument("--handshake", action="store_true", help="connect only, no audio")
    p.add_argument("--telephony", action="store_true", help="8kHz mu-law path (Twilio shape)")
    p.add_argument("--batch", action="store_true", help="open-weights path: OpenAI-compatible /v1/audio/transcriptions")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base for --batch")
    p.add_argument("--language", default="en", help='language code, or "auto" (default: en)')
    p.add_argument("--model", default=QWEN_ASR_DEFAULT_MODEL, help="model id")
    args = p.parse_args()

    # Harness convenience only — bolna itself stays vendor-neutral and reads
    # QWEN_ASR_BATCH_KEY / QWEN_API_KEY / DASHSCOPE_API_KEY.
    batch_keys = ("QWEN_ASR_BATCH_KEY", "OPENROUTER_API_KEY", "DEEPINFRA_API_KEY") if args.batch else ()
    api_key = (
        next((os.getenv(k) for k in batch_keys if os.getenv(k)), None)
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        if args.batch:
            print("[error] no key for the batch host. Set QWEN_ASR_BATCH_KEY, or export")
            print("        OPENROUTER_API_KEY / DEEPINFRA_API_KEY and this picks it up.")
        else:
            print("[error] set QWEN_API_KEY (or DASHSCOPE_API_KEY) — from Alibaba Cloud Model Studio")
        sys.exit(1)
    if not args.handshake and not os.path.exists(args.wav):
        print(f"[error] no such WAV: {args.wav}")
        sys.exit(1)

    if args.handshake and args.batch:
        print("[error] --handshake is realtime-only; the batch path has no session handshake")
        sys.exit(1)
    runner = run_handshake if args.handshake else run_full
    sys.exit(asyncio.run(runner(args, api_key)))


if __name__ == "__main__":
    main()
