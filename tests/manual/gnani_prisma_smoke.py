"""Exercise both Prisma versions over REST and streaming with known mono PCM16 WAV recordings.

From the repository root, with GNANI_API_KEY set:
    python -m tests.manual.gnani_prisma_smoke english.wav "hello world" hindi.wav "नमस्ते"
Each phrase must occur in its corresponding recording. Use single utterances under 30 seconds.
"""

import argparse
import asyncio
import audioop
import time
import wave
from pathlib import Path

from bolna.models import Transcriber
from bolna.providers import SUPPORTED_TRANSCRIBER_PROVIDERS


async def stream_turns(asr, path, telephony):
    with wave.open(str(path), "rb") as recording:
        if (recording.getsampwidth(), recording.getnchannels()) != (2, 1):
            raise ValueError("Provide a mono PCM16 WAV")
        pcm = recording.readframes(recording.getnframes())
        pcm, _ = audioop.ratecv(pcm, 2, 1, recording.getframerate(), asr.sampling_rate, None)
    clip = audioop.lin2ulaw(pcm, 2) if telephony else pcm
    bytes_per_second = asr.sampling_rate * (1 if telephony else 2)
    silence = b"\xff" if telephony else b"\0"
    audio = clip + silence * (2 * bytes_per_second) + clip
    chunk = bytes_per_second // 50
    await asr.run()
    try:
        for i in range(0, len(audio), chunk):
            await asr.input_queue.put({"data": audio[i : i + chunk], "meta_info": {"sequence": 0}})
            await asyncio.sleep(0.02)
        await asr.input_queue.put({"meta_info": {"eos": True}})
        await asyncio.wait_for(asr.transcription_task, 25)
        if asr.connection_error:
            raise RuntimeError(asr.connection_error)
        results = []
        while not asr.transcriber_output_queue.empty():
            packet = asr.transcriber_output_queue.get_nowait()
            data = packet["data"]
            if isinstance(data, dict) and data.get("type") == "transcript":
                results.append((data["content"], packet["meta_info"]))
        return results
    finally:
        await asr.cleanup()


async def main(args):
    failures = []
    for model, language, path, expected in (
        ("prisma-v2.5", "en-IN", args.english, args.english_phrase),
        ("prisma-v3.0", "indic-auto", args.hindi, args.hindi_phrase),
    ):
        for stream in (False, True):
            label = f"{model} {'WebSocket' if stream else 'REST'}"
            telephony = stream and language == "en-IN"
            config = Transcriber(provider="gnani", model=model, language=language, stream=stream)
            asr = SUPPORTED_TRANSCRIBER_PROVIDERS[config.provider](
                "twilio" if telephony else "default",
                input_queue=asyncio.Queue(),
                output_queue=asyncio.Queue(),
                **config.model_dump(),
            )
            started = time.monotonic()
            try:
                if stream:
                    results = await stream_turns(asr, path, telephony)
                else:
                    packet = await asr.transcribe_recording(path.read_bytes())
                    results = [(packet["data"], packet["meta_info"])]
                for text, meta in results:
                    print(
                        label,
                        repr(text),
                        meta.get("detected_language_code"),
                        "latency (s)",
                        meta.get("transcriber_latency"),
                    )
                if len(results) != (2 if stream else 1):
                    raise ValueError(f"received {len(results)} final transcripts")
                for text, meta in results:
                    if expected.casefold() not in text.casefold():
                        raise ValueError(f"missing expected phrase {expected!r}")
                    if language == "indic-auto" and meta.get("detected_language_code") != "hi-IN":
                        raise ValueError("expected hi-IN detection")
            except Exception as exc:
                failures.append(f"{label}: {exc}")
            print(f"{label} elapsed {time.monotonic() - started:.2f}s")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("english", type=Path)
    parser.add_argument("english_phrase")
    parser.add_argument("hindi", type=Path)
    parser.add_argument("hindi_phrase")
    asyncio.run(main(parser.parse_args()))
