"""A voice whose synthesizer fails moves to its next fallback provider and the turn in progress is sent to it;
with no fallback left the failure reaches the call instead of leaving it silent."""

import asyncio

import pytest

from bolna.helpers.fallbacks import FallbackEntry
from bolna.synthesizer.synthesizer_pool import SynthesizerPool


class FakeSynth:
    """Plays one audio packet per pushed text; `fail` makes the next generate() wait for it and raise."""

    def __init__(self, model, characters=0):
        self.model = model
        self.pushed = []
        self.audio = asyncio.Queue()
        self.monitor_starts = 0
        self.cleaned_up = False
        self.characters = characters
        self.turn_latencies = []

    async def monitor_connection(self):
        self.monitor_starts += 1

    async def push(self, message):
        self.pushed.append(message["data"])
        await self.audio.put(message)

    def fail(self, error):
        self.audio.put_nowait(error)

    async def generate(self):
        while True:
            message = await self.audio.get()
            if isinstance(message, Exception):
                raise message
            yield {"data": f"{self.model}:{message['data']}".encode(), "meta_info": message["meta_info"]}

    def get_synthesized_characters(self):
        return self.characters

    async def handle_interruption(self):
        pass

    async def cleanup(self):
        self.cleaned_up = True


def text(chunk, sequence_id=1, last=False):
    return {"data": chunk, "meta_info": {"sequence_id": sequence_id, "end_of_llm_stream": last}}


async def started_pool(primary, *fallbacks):
    switches = []
    pool = SynthesizerPool(
        {"hi": primary},
        "hi",
        {},
        lazy_legs=True,
        fallbacks={"hi": [FallbackEntry("sarvam", synth.model, lambda synth=synth: synth) for synth in fallbacks]},
        on_fallback=lambda label, failed, entry, error: switches.append((failed.model, entry.model, str(error))),
    )
    await pool.monitor_connection()
    return pool, switches


async def next_audio(pool, stream):
    return (await asyncio.wait_for(stream.__anext__(), timeout=1))["data"]


async def test_a_failure_mid_turn_has_the_fallback_speak_the_whole_turn_again():
    primary, backup = FakeSynth("sonic-3"), FakeSynth("bulbul:v3")
    pool, switches = await started_pool(primary, backup)
    stream = pool.generate()

    await pool.push(text("Namaste,"))
    assert await next_audio(pool, stream) == b"sonic-3:Namaste,"
    primary.fail(RuntimeError("HTTP 429 too many connections"))
    await pool.push(text(" kaise madad karun?"))

    assert await next_audio(pool, stream) == b"bulbul:v3:Namaste,"
    assert await next_audio(pool, stream) == b"bulbul:v3: kaise madad karun?"
    assert switches == [("sonic-3", "bulbul:v3", "HTTP 429 too many connections")]
    assert backup.monitor_starts == 1
    await asyncio.sleep(0)
    assert primary.cleaned_up
    await pool.cleanup()


async def test_a_finished_turn_is_not_spoken_again_by_the_fallback():
    primary, backup = FakeSynth("sonic-3"), FakeSynth("bulbul:v3")
    pool, _ = await started_pool(primary, backup)
    stream = pool.generate()

    await pool.push(text("Theek hai.", last=True))
    packet = await asyncio.wait_for(stream.__anext__(), timeout=1)
    packet["meta_info"]["end_of_synthesizer_stream"] = True
    pool.settle_open_turn(packet)
    primary.fail(RuntimeError("socket dropped"))
    await pool.push(text("Aur kuch?", sequence_id=2))

    assert await next_audio(pool, stream) == b"bulbul:v3:Aur kuch?"
    assert backup.pushed == ["Aur kuch?"]
    await pool.cleanup()


async def test_with_no_fallback_left_the_failure_reaches_the_call():
    primary = FakeSynth("sonic-3")
    pool, _ = await started_pool(primary)
    stream = pool.generate()

    primary.fail(RuntimeError("HTTP 429 too many connections"))
    with pytest.raises(RuntimeError, match="429"):
        await asyncio.wait_for(stream.__anext__(), timeout=1)
    await pool.cleanup()


async def test_the_failed_voices_characters_still_count_toward_usage():
    primary, backup = FakeSynth("sonic-3", characters=40), FakeSynth("bulbul:v3", characters=25)
    pool, _ = await started_pool(primary, backup)
    stream = pool.generate()
    primary.fail(RuntimeError("socket dropped"))
    await pool.push(text("Ji."))
    await next_audio(pool, stream)

    assert pool.get_synthesized_characters() == 65
    await pool.cleanup()
