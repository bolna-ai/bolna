"""Check that ElevenLabs does not say a word twice where it splits a reply.

Left to split a reply itself, ElevenLabs can cut right after a linking word such as "और" and say
it again at the start of the next piece. Where it cuts depends on the voice. Each reply is sent two
ways: "unbuffered" (every word straight away and one flush at the end, as the sender used to) and
"sentence" (the current sender, which flushes at every sentence end), and each render is checked
for a repeated "और". The repeat is intermittent, so use several runs.

Exits non-zero if the current sender repeated the word.
"""

import argparse
import asyncio
import audioop
import io
import json
import logging
import os
import re
import sys
import time
import wave
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bolna.synthesizer.elevenlabs_synthesizer import ElevenlabsSynthesizer  # noqa: E402

WORD = "और"

CASES = {
    # A sentence opening with "और", pushed in pieces the way the LLM streams it.
    "after-sentence-end": [
        "हम twenty years से भी ज्यादा समय से आपकी सेवा कर रहे हैं, तो आप बिल्कुल सही हाथों में हैं। और Smart Saver Plan",
        "हमारे सबसे popular plans में से एक है, पिछले महीने four thousand लोगों ने इसे चुना है।",
        "क्या मैं आपको इस plan के बारे में कुछ information बताऊँ?",
    ],
    # "और" mid-sentence, with all three sentences in one push.
    "mid-sentence": [
        (
            "Perfect, मैंने checkout link आपके WhatsApp पर फिर से भेज दिया है। वैसे, हम आपकी fitness journey में और "
            "कैसे help कर सकते हैं। आपका कोई और question हो जिसमें मैं help कर सकूँ?"
        ),
    ],
}
VARIANTS = ("unbuffered", "sentence")


class _Pipeline:
    """Stands in for the task manager: every sequence is live."""

    def __init__(self):
        self.conversation_start_init_ts = time.time() * 1000

    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


def _message(text, end_of_llm_stream):
    return {
        "data": text,
        "meta_info": {
            "sequence_id": 1,
            "turn_id": 1,
            "end_of_llm_stream": end_of_llm_stream,
            "tts_start_ms": 0,
            "message_category": None,
            "request_id": "seam-repeat",
        },
    }


def count_word(text):
    return sum(1 for w in re.split(r"[\s,।.?!;:\"'()\-]+", text) if w == WORD)


def to_wav(ulaw):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(audioop.ulaw2lin(ulaw, 2))
    return buf.getvalue()


async def send_unbuffered(synth, pushes):
    """The sender before sentence flushing: every word straight away, one flush at the end."""
    synth._on_push({"sequence_id": 1}, "")
    synth.ws_send_time = time.perf_counter()
    for text in pushes:
        for chunk in synth.text_chunker(text):
            await synth.websocket.send(json.dumps({"text": chunk, "context_id": synth.context_id}))
    await synth.websocket.send(json.dumps({"text": "", "context_id": synth.context_id, "flush": True}))
    await synth.websocket.send(json.dumps({"context_id": synth.context_id, "close_context": True}))


async def send_through_sender(synth, pushes):
    for text in pushes:
        await synth.push(_message(text, False))
        await synth.sender_task
    await synth.push(_message("", True))
    await synth.sender_task


async def render(args, pushes, variant):
    """Synthesize one turn on a fresh socket and return the (audio, aligned_text) frames."""
    synth = ElevenlabsSynthesizer(
        voice="voice",
        voice_id=args.voice_id,
        model=args.model,
        synthesizer_key=args.key,
        task_manager_instance=_Pipeline(),
        **json.loads(args.config or "{}"),
    )
    synth.websocket = await synth.establish_connection()
    if synth.websocket is None:
        raise RuntimeError(f"connection failed: {synth.connection_error}")

    frames = []

    async def consume():
        async for audio, aligned in synth.receiver():
            if audio == b"\x00":
                return
            frames.append((audio, aligned))

    consumer = asyncio.create_task(consume())
    try:
        if variant == "unbuffered":
            await send_unbuffered(synth, pushes)
        else:
            await send_through_sender(synth, pushes)
        await asyncio.wait_for(consumer, timeout=45)
    finally:
        consumer.cancel()
        await synth.cleanup()
    return frames


async def transcribe(session, args, wav):
    form = aiohttp.FormData()
    form.add_field("model_id", args.stt_model)
    form.add_field("language_code", "hi")
    form.add_field("file", wav, filename="render.wav", content_type="audio/wav")
    url = f"https://{os.getenv('ELEVENLABS_API_HOST', 'api.elevenlabs.io')}/v1/speech-to-text"
    async with session.post(url, headers={"xi-api-key": args.key}, data=form) as resp:
        body = await resp.json(content_type=None)
        if resp.status != 200:
            raise RuntimeError(f"speech-to-text {resp.status}: {body}")
        return body["text"]


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES), action="append", help="default: all")
    parser.add_argument("--variant", choices=VARIANTS, action="append", help="default: all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--voice-id", default="OtEfb2LVzIE45wdYe54M")
    parser.add_argument("--model", default="eleven_flash_v2_5")
    parser.add_argument("--config", help="extra synthesizer fields as JSON, e.g. temperature/similarity_boost")
    parser.add_argument("--stt-model", default="scribe_v1")
    parser.add_argument("--no-transcribe", action="store_true")
    parser.add_argument("--key", default=os.getenv("ELEVENLABS_API_KEY"))
    parser.add_argument("--keep-audio", help="directory to write each render's WAV into")
    parser.add_argument("--verbose", action="store_true", help="keep the synthesizer's INFO logs")
    args = parser.parse_args()
    if not args.key:
        sys.exit("no API key: pass --key or set ELEVENLABS_API_KEY")
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
    keep = Path(args.keep_audio) if args.keep_audio else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)

    tally = {}
    async with aiohttp.ClientSession() as session:
        for case in args.case or sorted(CASES):
            pushes = CASES[case]
            expected = count_word(" ".join(pushes))
            for variant in args.variant or VARIANTS:
                for run in range(1, args.runs + 1):
                    frames = await render(args, pushes, variant)
                    audio = b"".join(a for a, _ in frames)
                    wav = to_wav(audio)
                    if keep:
                        (keep / f"{case}_{variant}_{run}.wav").write_bytes(wav)

                    ends_on_word = any(aligned.strip().endswith(WORD) for _, aligned in frames[:-1])
                    heard, transcript = None, ""
                    if not args.no_transcribe:
                        try:
                            transcript = await transcribe(session, args, wav)
                            heard = count_word(transcript)
                        except Exception as e:
                            transcript = f"(transcription failed: {e})"
                    repeated = heard is not None and heard > expected

                    verdict = "REPEAT" if repeated else ("ok" if heard is not None else "?")
                    print(
                        f"{case:<18} {variant:<10} run{run}  {WORD} {heard}/{expected}  {verdict:<6} "
                        f"{len(audio) / 8000:5.1f}s  frame-ends-on-{WORD}={ends_on_word}"
                    )
                    print("    frames: " + " | ".join(f"{t!r} {len(a) / 8000:.2f}s" for a, t in frames))
                    if transcript:
                        print(f"    heard: {transcript}")
                    stats = tally.setdefault((case, variant), [0, 0])
                    stats[0] += repeated
                    stats[1] += 1

    print()
    for (case, variant), (repeats, runs) in tally.items():
        print(f"{case:<18} {variant:<10} {repeats}/{runs} renders repeated {WORD}")
    sentence_repeats = sum(r for (_, v), (r, _) in tally.items() if v == "sentence")
    return 1 if sentence_repeats else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
