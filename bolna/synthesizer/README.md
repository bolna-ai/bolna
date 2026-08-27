# Adding a TTS provider

Four registrations, one synthesizer class, one smoke run.

## Register

1. `bolna/enums.py`: add the slug to `SynthesizerProvider`.
2. `bolna/models.py`: add a config class, add it to `Synthesizer.provider_config`'s union, and add one
   entry to `SYNTHESIZER_CONFIG_MODELS`. Extend `StandardVoiceConfig` unless the provider genuinely
   needs a different shape: a config of `voice`, `voice_id`, `model` and `language` needs no
   per-provider code downstream to build agent configs for it.
3. `bolna/synthesizer/<name>_synthesizer.py`: the class. Subclass `StreamSynthesizer` for a websocket
   provider, `BaseSynthesizer` for an HTTP-only one.
4. `bolna/synthesizer/__init__.py` and `bolna/providers.py`: export it and register it in
   `SUPPORTED_SYNTHESIZER_MODELS`.

## The contract

`StreamSynthesizer` owns push routing, latency bookkeeping, reconnects and cleanup. A subclass
supplies `establish_connection`, `sender`, `receiver`, and usually `_process_audio_chunk`.

- **One end-of-stream sentinel per turn.** `receiver()` yields `b"\x00"` exactly once, when the
  provider reports the utterance complete. A second one stamps end-of-stream onto the next turn; a
  missing one leaves the turn hanging until the pipeline's timeout.
- **A cancelled turn gets no sentinel.** `handle_interruption()` has already abandoned it, so its
  terminator must not be forwarded.
- **Drop in-flight audio after a barge-in.** Frames generated before the provider processed the
  cancel keep arriving. Anything still forwarded is popped against the next turn's metadata and
  plays as the start of the new reply, so track the provider's response or context id and discard
  by it.
- **The turn's final LLM chunk is often empty.** `end_of_llm_stream` rides on it, so flush on the
  flag rather than on having text.
- **`synthesize()` returns a self-describing container.** Callers convert it for telephony with a
  rate hint guessed from the synthesizer, which headerless PCM gets wrong.
- **A failed one-shot render returns `None`.** The shared HTTP loop turns that into an end-of-turn;
  never yield a `None` audio packet.
- **`voice` is never `None`.** The platform reads it at call setup.

## Verify

```bash
<PROVIDER>_API_KEY=... python3 tests/manual/tts_smoke.py --provider <slug> --voice <name>
```

Runs a streamed turn, a turn whose final chunk is empty, a barge-in, back-to-back turns and both
one-shot paths against the live API. Every check should pass before the PR goes up.
