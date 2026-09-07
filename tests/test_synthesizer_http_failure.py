"""A one-shot render that fails must end the turn, not hand the output handler a None packet."""

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from bolna.synthesizer.base_synthesizer import BaseSynthesizer


class _AlwaysCurrent:
    def is_sequence_id_in_current_ids(self, sequence_id):
        return True


class _HttpSynth(BaseSynthesizer):
    """Renders through the shared HTTP loop; the provider call is scripted per test."""

    def __init__(self, audio):
        super().__init__(task_manager_instance=_AlwaysCurrent(), stream=False)
        self._audio = audio
        self.calls = 0
        self.caching = True
        self.cache = InmemoryScalarCache()

    async def _generate_http(self, text):
        self.calls += 1
        return self._audio


async def _first_packet(synth, text="hello"):
    await synth.push({"data": text, "meta_info": {"sequence_id": 1, "end_of_llm_stream": True}})
    return await synth._generate_http_loop().__anext__()


async def test_a_failed_render_yields_the_end_of_stream_sentinel():
    synth = _HttpSynth(None)
    packet = await _first_packet(synth)
    assert packet["data"] == b"\x00"
    assert packet["meta_info"]["end_of_synthesizer_stream"] is True


async def test_a_failed_render_is_not_cached():
    synth = _HttpSynth(None)
    await _first_packet(synth)
    assert synth.cache.get("hello") is None


async def test_a_successful_render_is_passed_through_and_cached():
    synth = _HttpSynth(b"audio-bytes")
    packet = await _first_packet(synth)
    assert packet["data"] == b"audio-bytes"
    assert synth.cache.get("hello") == b"audio-bytes"
