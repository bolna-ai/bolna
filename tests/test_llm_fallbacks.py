"""A conversation turn that fails before any output moves to the next fallback model, which serves the rest of the
call; a turn that already produced output is never re-run."""

import pytest

from bolna.helpers.fallbacks import FallbackEntry
from bolna.llms.fallback_llm import FallbackLLM, fallback_llm_arguments


class ScriptedLLM:
    """Each turn either raises before output, raises after its first chunk, or streams its chunks."""

    def __init__(self, model, *turns):
        self.model = model
        self.turns = list(turns)
        self.calls = 0
        self.closed = False
        self.fails_fast = False

    def fail_fast(self):
        self.fails_fast = True

    async def generate_stream(self, messages, synthesize=True, meta_info=None):
        self.calls += 1
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        for chunk in turn:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def invalidate_response_chain(self):
        self.invalidated = True

    async def close(self):
        self.closed = True


def entry(llm, builds=None):
    model = getattr(llm, "model", "broken")

    def build():
        if builds is not None:
            builds.append(model)
        if isinstance(llm, Exception):
            raise llm
        return llm

    return FallbackEntry("azure", model, build)


async def turn(chain):
    return [chunk async for chunk in chain.generate_stream([{"role": "user", "content": "hi"}], meta_info={})]


async def test_failure_before_output_reruns_the_turn_on_the_next_entry_and_stays_there():
    primary = ScriptedLLM("primary", RuntimeError("429 no capacity"))
    backup = ScriptedLLM("backup", ["Sure."], ["Done."])
    switches = []
    chain = FallbackLLM(primary, [entry(backup)], on_switch=lambda error, e, meta: switches.append(e.model))

    assert await turn(chain) == ["Sure."]
    assert await turn(chain) == ["Done."]

    assert primary.calls == 1
    assert switches == ["backup"]
    assert chain.model == "backup"


async def test_failure_after_output_is_raised_rather_than_repeating_the_reply():
    primary = ScriptedLLM("primary", ["Let me check", RuntimeError("stream cut")])
    backup = ScriptedLLM("backup", ["never"])
    chain = FallbackLLM(primary, [entry(backup)])

    with pytest.raises(RuntimeError, match="stream cut"):
        await turn(chain)
    assert backup.calls == 0


async def test_a_failing_backup_walks_on_and_the_last_error_ends_the_chain():
    primary = ScriptedLLM("primary", RuntimeError("429"))
    second = ScriptedLLM("second", RuntimeError("503"))
    third = ScriptedLLM("third", RuntimeError("timeout"))
    chain = FallbackLLM(primary, [entry(second), entry(third)])

    with pytest.raises(RuntimeError, match="timeout"):
        await turn(chain)
    assert (primary.calls, second.calls, third.calls) == (1, 1, 1)


async def test_backups_are_built_only_when_needed_and_one_that_cannot_build_is_skipped():
    builds = []
    primary = ScriptedLLM("primary", ["Hello."], RuntimeError("429"))
    backup = ScriptedLLM("backup", ["Hi again."])
    chain = FallbackLLM(primary, [entry(ValueError("no such deployment"), builds), entry(backup, builds)])

    assert await turn(chain) == ["Hello."]
    assert builds == []

    assert await turn(chain) == ["Hi again."]
    assert builds == ["broken", "backup"]


async def test_only_the_last_model_retries_a_failed_request_in_place():
    primary = ScriptedLLM("primary", RuntimeError("429"))
    second = ScriptedLLM("second", RuntimeError("429"))
    last = ScriptedLLM("last", ["ok"])
    chain = FallbackLLM(primary, [entry(second), entry(last)])
    await turn(chain)

    assert (primary.fails_fast, second.fails_fast, last.fails_fast) == (True, True, False)


async def test_close_releases_every_llm_the_call_used():
    primary = ScriptedLLM("primary", RuntimeError("429"))
    backup = ScriptedLLM("backup", ["ok"])
    chain = FallbackLLM(primary, [entry(backup)])
    await turn(chain)

    await chain.close()
    assert primary.closed and backup.closed


def test_an_entry_keeps_the_agents_settings_but_never_the_primarys_backend_or_model_settings():
    llm_config = {
        "provider": "azure",
        "model": "azure/gpt-5.4-mini",
        "max_tokens": 300,
        "temperature": 0.2,
        "reasoning_effort": "minimal",
    }
    kwargs = {"llm_key": "azure-key", "base_url": "https://in.azure", "overflow_llm": {"model": "x"}, "run_id": "r"}
    entry_config = {"provider": "openai", "model": "gpt-4.1-mini", "reasoning_effort": None}

    config, entry_kwargs = fallback_llm_arguments(llm_config, kwargs, entry_config, {"llm_key": "openai-key"})

    assert config == {"provider": "openai", "model": "gpt-4.1-mini", "max_tokens": 300, "temperature": 0.2}
    assert entry_kwargs == {"run_id": "r", "llm_key": "openai-key"}
