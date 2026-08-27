"""Drive a TTS provider through the synthesizer contract against its live API.

Checks the behaviours a synthesizer is easy to get wrong: one end-of-stream sentinel per turn,
a turn whose final LLM chunk is empty still flushing, no audio surviving a barge-in, and the
one-shot HTTP renders the welcome/handoff paths depend on.

Usage:
    KALPA_API_KEY=... python3 tests/manual/tts_smoke.py --provider kalpa --voice Kiara
    ELEVENLABS_API_KEY=... python3 tests/manual/tts_smoke.py --provider elevenlabs \
        --voice George --voice-id JBFqnCBsd6RMkjVDRZzb --model eleven_turbo_v2_5
    ... --config '{"chunk_length_schedule": [50, 80]}' --web --keep-audio out/
"""

import argparse
import asyncio
import audioop
import json
import os
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS  # noqa: E402

# Long enough that generation is still running when the barge-in lands.
LONG_TEXT = (
    "Let me walk you through the whole process, starting with account verification, then the "
    "billing history, then the plan options, and finally the confirmation email you will receive."
)
# Split the way the streaming LLM wrappers do: rsplit(" ", 1) consumes the boundary space, so a
# synthesizer that concatenates verbatim glues the words together.
TURN_CHUNKS = ["Thanks for calling. I have pulled up your account and I can", "see the last payment cleared."]


class _Pipeline:
    """Stands in for the task manager: owns which sequence is live, nothing else."""

    def __init__(self):
        self.live = set()
        self.conversation_start_init_ts = time.time() * 1000

    def is_sequence_id_in_current_ids(self, sequence_id):
        return sequence_id in self.live


class Smoke:
    def __init__(self, synth, pipeline, keep_audio=None):
        self.synth = synth
        self.pipeline = pipeline
        self.keep_audio = Path(keep_audio) if keep_audio else None
        self.packets = []
        self.results = []
        self._tasks = []

    async def start(self):
        if self.synth.stream:
            self._tasks.append(asyncio.create_task(self.synth.monitor_connection()))
            deadline = time.perf_counter() + 30
            while not self.synth._is_ws_connected():
                if self.synth.connection_error:
                    raise RuntimeError(f"connection rejected: {self.synth.connection_error}")
                if time.perf_counter() > deadline:
                    raise RuntimeError("synthesizer never connected")
                await asyncio.sleep(0.05)
        self._tasks.append(asyncio.create_task(self._consume()))

    async def _consume(self):
        async for packet in self.synth.generate():
            self.packets.append((time.perf_counter(), packet))

    async def stop(self):
        await self.synth.cleanup()
        for task in self._tasks:
            task.cancel()

    def record(self, name, ok, detail):
        self.results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {detail}")

    @staticmethod
    def _message(text, sequence_id, end_of_llm_stream):
        return {
            "data": text,
            "meta_info": {
                "sequence_id": sequence_id,
                "turn_id": sequence_id,
                "end_of_llm_stream": end_of_llm_stream,
                "tts_start_ms": 0,
                "message_category": None,
                "request_id": "tts-smoke",
            },
        }

    async def turn(self, sequence_id, chunks, gap=0.15, timeout=45):
        """Push a turn the way the pipeline does and wait for its end-of-stream sentinel."""
        self.pipeline.live.add(sequence_id)
        first_index = len(self.packets)
        started = time.perf_counter()
        for chunk in chunks:
            await self.synth.push(self._message(chunk, sequence_id, False))
            await asyncio.sleep(gap)
        flushed = time.perf_counter()
        await self.synth.push(self._message("", sequence_id, True))

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if any(p["meta_info"].get("end_of_synthesizer_stream") for _, p in self._mine(first_index, sequence_id)):
                break
            await asyncio.sleep(0.02)

        mine = self._mine(first_index, sequence_id)
        audio = [(t, p) for t, p in mine if p["data"] != b"\x00"]
        return {
            "sequence_id": sequence_id,
            "settled": any(p["meta_info"].get("end_of_synthesizer_stream") for _, p in mine),
            "packets": len(audio),
            "seconds": round(sum(self._seconds(p) for _, p in audio), 2),
            "ttfb_ms": round((audio[0][0] - started) * 1000) if audio else None,
            "before_flush_ms": round((flushed - audio[0][0]) * 1000) if audio else None,
            "audio": b"".join(p["data"] for _, p in audio),
            "format": mine[0][1]["meta_info"].get("format") if mine else None,
        }

    def _mine(self, first_index, sequence_id):
        return [(t, p) for t, p in self.packets[first_index:] if p["meta_info"].get("sequence_id") == sequence_id]

    def _rate(self):
        # Providers carry this as either an int or a string, depending on how the config declares it.
        return int(self.synth.sampling_rate or 8000)

    def _seconds(self, packet):
        width = 1 if packet["meta_info"].get("format") == "mulaw" else 2
        return len(packet["data"]) / (self._rate() * width)

    def save(self, name, turn):
        if not self.keep_audio or not turn["audio"]:
            return
        self.keep_audio.mkdir(parents=True, exist_ok=True)
        pcm = audioop.ulaw2lin(turn["audio"], 2) if turn["format"] == "mulaw" else turn["audio"]
        with wave.open(str(self.keep_audio / f"{name}.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self._rate())
            out.writeframes(pcm)

    async def check_streaming_turn(self):
        turn = await self.turn(1, TURN_CHUNKS)
        self.save("streamed_turn", turn)
        detail = f"{turn['seconds']}s in {turn['packets']} packets, ttfb {turn['ttfb_ms']}ms"
        if turn["before_flush_ms"] and turn["before_flush_ms"] > 0:
            detail += f", first audio {turn['before_flush_ms']}ms before the flush"
        self.record("streamed turn settles", turn["settled"] and turn["packets"] > 0, detail)

    async def check_empty_final_chunk(self):
        """The turn's last LLM buffer is often empty; end_of_llm_stream rides on it."""
        turn = await self.turn(2, ["One moment please."])
        self.save("empty_final_chunk", turn)
        self.record(
            "empty final chunk flushes",
            turn["settled"] and turn["packets"] > 0,
            f"{turn['seconds']}s, settled={turn['settled']}",
        )

    async def check_barge_in(self):
        self.pipeline.live.add(3)
        first_index = len(self.packets)
        await self.synth.push(self._message(LONG_TEXT, 3, False))
        await self.synth.push(self._message("", 3, True))

        deadline = time.perf_counter() + 25
        while time.perf_counter() < deadline and not self._mine(first_index, 3):
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)

        self.pipeline.live.discard(3)
        interrupted_at = time.perf_counter()
        await self.synth.handle_interruption()
        await asyncio.sleep(2.0)

        leaked = [t for t, p in self.packets if t > interrupted_at and p["data"] != b"\x00"]
        self.record("barge-in drops in-flight audio", not leaked, f"{len(leaked)} audio packets after the cancel")

        recovery = await self.turn(4, ["Sure, no problem."])
        self.save("after_barge_in", recovery)
        self.record(
            "turn after barge-in is clean",
            recovery["settled"] and recovery["packets"] > 0,
            f"{recovery['seconds']}s, ttfb {recovery['ttfb_ms']}ms",
        )

    async def check_back_to_back(self):
        first = await self.turn(5, ["First short reply, done."])
        second = await self.turn(6, ["Second short reply, done."])
        self.record(
            "back-to-back turns settle",
            first["settled"] and second["settled"],
            f"ttfb {first['ttfb_ms']}ms then {second['ttfb_ms']}ms",
        )

    async def check_one_shot(self):
        audio = await self.synth.synthesize("Please hold while I transfer your call.")
        self.record("synthesize() returns audio", bool(audio), f"{len(audio or b'')} bytes")

        clip = await self.synth.synthesize_telephony_clip("Please hold while I transfer your call.")
        if clip is None:
            self.record("telephony one-shot", True, "not offered; caller falls back to synthesize()")
        else:
            self.record("telephony one-shot", bool(clip), f"{round(len(clip) / 8000, 2)}s of mu-law")


def build_synthesizer(args):
    provider_config = json.loads(args.config) if args.config else {}
    for field, value in (
        ("voice", args.voice),
        ("voice_id", args.voice_id),
        ("model", args.model),
        ("language", args.language),
    ):
        if value is not None:
            provider_config[field] = value

    key = args.key or os.getenv(f"{args.provider.upper()}_API_KEY")
    if not key:
        sys.exit(f"no API key: pass --key or set {args.provider.upper()}_API_KEY")

    pipeline = _Pipeline()
    synth = SUPPORTED_SYNTHESIZER_MODELS[args.provider](
        **provider_config,
        stream=not args.no_stream,
        use_mulaw=not args.web,
        synthesizer_key=key,
        task_manager_instance=pipeline,
    )
    return synth, pipeline


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, choices=sorted(SUPPORTED_SYNTHESIZER_MODELS))
    parser.add_argument("--voice")
    parser.add_argument("--voice-id")
    parser.add_argument("--model")
    parser.add_argument("--language")
    parser.add_argument("--config", help="extra provider_config fields as JSON")
    parser.add_argument("--key", help="defaults to <PROVIDER>_API_KEY")
    parser.add_argument("--web", action="store_true", help="render PCM for web instead of telephony mu-law")
    parser.add_argument("--no-stream", action="store_true", help="exercise the one-shot HTTP path only")
    parser.add_argument("--keep-audio", help="directory to write each turn's WAV into")
    args = parser.parse_args()

    synth, pipeline = build_synthesizer(args)
    smoke = Smoke(synth, pipeline, args.keep_audio)
    print(f"provider={args.provider} stream={synth.stream} mulaw={getattr(synth, 'use_mulaw', False)}")

    await smoke.start()
    try:
        if synth.stream:
            print(f"connected in {synth.connection_time}ms")
            await smoke.check_streaming_turn()
            await smoke.check_empty_final_chunk()
            await smoke.check_barge_in()
            await smoke.check_back_to_back()
        await smoke.check_one_shot()
    finally:
        await smoke.stop()

    failed = [name for name, ok, _ in smoke.results if not ok]
    print(f"\n{len(smoke.results) - len(failed)}/{len(smoke.results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
