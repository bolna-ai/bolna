"""A conversation turn that fails before any output moves to the next fallback model, which serves the rest of the
call; a turn that already produced output is never re-run."""

import pytest

from bolna.helpers.fallbacks import FallbackEntry
from bolna.llms.fallback_llm import FallbackLLM, fallback_llm_arguments


class Refused(Exception):
    """A capacity or config rejection, which no retry on the same model fixes."""

    status_code = 429


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
    primary = ScriptedLLM("primary", Refused("429 no capacity"))
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
    primary = ScriptedLLM("primary", Refused("429"))
    second = ScriptedLLM("second", Refused("503"))
    third = ScriptedLLM("third", Refused("timeout"))
    chain = FallbackLLM(primary, [entry(second), entry(third)])

    with pytest.raises(Refused, match="timeout"):
        await turn(chain)
    assert (primary.calls, second.calls, third.calls) == (1, 1, 1)


async def test_backups_are_built_only_when_needed_and_one_that_cannot_build_is_skipped():
    builds = []
    primary = ScriptedLLM("primary", ["Hello."], Refused("429"))
    backup = ScriptedLLM("backup", ["Hi again."])
    chain = FallbackLLM(primary, [entry(ValueError("no such deployment"), builds), entry(backup, builds)])

    assert await turn(chain) == ["Hello."]
    assert builds == []

    assert await turn(chain) == ["Hi again."]
    assert builds == ["broken", "backup"]


async def test_only_the_last_model_retries_a_failed_request_in_place():
    primary = ScriptedLLM("primary", Refused("429"))
    second = ScriptedLLM("second", Refused("429"))
    last = ScriptedLLM("last", ["ok"])
    chain = FallbackLLM(primary, [entry(second), entry(last)])
    await turn(chain)

    assert (primary.fails_fast, second.fails_fast, last.fails_fast) == (True, True, False)


async def test_close_releases_every_llm_the_call_used():
    primary = ScriptedLLM("primary", Refused("429"))
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


async def test_two_turns_failing_together_move_to_the_same_fallback_once():
    import asyncio

    class SlowToFail(ScriptedLLM):
        async def generate_stream(self, messages, synthesize=True, meta_info=None):
            self.calls += 1
            await asyncio.sleep(0)
            raise Refused("429 no capacity")
            yield

    primary = SlowToFail("primary")
    backup = ScriptedLLM("backup", ["Sure."], ["Done."])
    unused = ScriptedLLM("unused")
    chain = FallbackLLM(primary, [entry(backup), entry(unused)])

    assert sorted(await asyncio.gather(turn(chain), turn(chain))) == [["Done."], ["Sure."]]
    assert chain.active is backup
    assert unused.calls == 0


async def test_a_passing_fault_gets_one_more_try_on_the_same_model_before_the_call_moves_off_it():
    primary = ScriptedLLM("primary", ConnectionError("reset"), ["Sure."], ConnectionError("reset"), ConnectionError())
    backup = ScriptedLLM("backup", ["Done."])
    retries = []
    chain = FallbackLLM(primary, [entry(backup)], on_retry=retries.append)

    assert await turn(chain) == ["Sure."]
    assert chain.active is primary
    # The failed attempt is reported, and its retry resends the whole conversation.
    assert (len(retries), primary.invalidated) == (1, True)

    assert await turn(chain) == ["Done."]
    assert (primary.calls, chain.active) == (4, backup)


async def test_a_fault_that_is_not_a_lost_connection_or_a_server_error_is_never_retried():
    for fault in (ValueError("context length exceeded"), KeyError("choices"), Refused("429")):
        primary = ScriptedLLM("primary", fault)
        chain = FallbackLLM(primary, [entry(ScriptedLLM("backup", ["Done."]))])
        assert await turn(chain) == ["Done."]
        assert primary.calls == 1


async def test_a_fault_that_took_long_to_come_back_moves_on_without_a_second_try(monkeypatch):
    import bolna.llms.fallback_llm as module

    monkeypatch.setattr(module, "RETRY_WITHIN_S", -1)
    primary = ScriptedLLM("primary", ConnectionError("timed out"))
    chain = FallbackLLM(primary, [entry(ScriptedLLM("backup", ["Done."]))])

    assert await turn(chain) == ["Done."]
    assert primary.calls == 1


async def test_the_agents_own_model_passed_over_at_the_start_is_closed_with_the_rest():
    own, serving = ScriptedLLM("own"), ScriptedLLM("serving", ["Sure."])
    chain = FallbackLLM(serving, [FallbackEntry("azure", "own", lambda: own)], passed_over=own)

    assert await turn(chain) == ["Sure."]
    await chain.close()

    assert own.closed and serving.closed
    assert not own.fails_fast
