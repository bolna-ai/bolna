"""Render audible WAV files for both Timbre models through Bolna using a live GNANI_API_KEY.

Run from the repository root: python -m tests.manual.gnani_timbre_smoke /tmp/timbre-audio
"""

import argparse
import asyncio
import audioop
import io
import wave
from contextlib import aclosing
from pathlib import Path

from bolna.models import Synthesizer
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS


async def render(model, voice, language, text, transport, destination):
    config = Synthesizer(provider="gnani", provider_config={"model": model, "voice": voice, "language": language})
    synth = SUPPORTED_SYNTHESIZER_MODELS[config.provider](
        **dict(config.provider_config.model_dump(), stream_transport="sse" if transport == "sse" else "websocket"),
        stream=transport != "rest",
        use_mulaw=transport == "websocket",
    )
    path = destination / f"{model}-{transport}.wav"
    try:
        if transport == "rest":
            wav = await synth.synthesize(text)
            with wave.open(io.BytesIO(wav), "rb") as reader:
                assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (
                    1,
                    2,
                    synth.sampling_rate,
                )
                pcm = reader.readframes(reader.getnframes())
        else:
            await synth.push({"data": text, "meta_info": {"sequence_id": 1, "end_of_llm_stream": True}})
            frames = []
            async with aclosing(synth.generate()) as packets:
                async for packet in packets:
                    meta = packet["meta_info"]
                    assert meta["sequence_id"] == 1 and meta["format"] == (
                        "mulaw" if transport == "websocket" else "pcm"
                    )
                    if packet["data"] != b"\x00":
                        frames.append(packet["data"])
                    if meta.get("end_of_synthesizer_stream"):
                        break
            pcm = b"".join(frames)
            if transport == "websocket":
                pcm = audioop.ulaw2lin(pcm, 2)
        assert len(pcm) > synth.sampling_rate and audioop.rms(pcm, 2) > 40, "empty, silent or truncated recording"
        with wave.open(str(path), "wb") as out:
            out.setparams((1, 2, synth.sampling_rate, 0, "NONE", "not compressed"))
            out.writeframes(pcm)
        print(f"{path.name}: {synth.sampling_rate} Hz, {len(pcm) / (2 * synth.sampling_rate):.2f}s", flush=True)
    finally:
        await synth.cleanup()


async def main(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for args in [
        ("timbre-v2.5", "Kaveri", "en-IN", "Welcome to Bolna. Your appointment is confirmed for Monday morning."),
        ("timbre-v3.0", "Shivani", "hi-IN", "नमस्ते। आपकी यात्रा का टिकट तैयार है। आपका दिन शुभ हो।"),
    ]:
        for transport in ["rest", "sse", "websocket"]:
            await asyncio.wait_for(render(*args, transport, destination), timeout=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    asyncio.run(main(parser.parse_args().destination))
