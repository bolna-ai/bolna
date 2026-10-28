"""An agent's fallbacks are wired at call setup: a single-provider transcriber or voice with fallbacks runs as a
one-language pool, each entry is built only when needed and never with the primary provider's key, and a switch
is recorded against the provider that failed."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager
from bolna.llms.fallback_llm import FallbackLLM


def bind(tm, *names):
    for name in names:
        setattr(tm, name, getattr(TaskManager, name).__get__(tm, TaskManager))


def recorder(calls, provider):
    def factory(*args, **kwargs):
        calls.append((provider, kwargs))
        return MagicMock(model=kwargs.get("model"))

    return factory


def setup_tm(task_config, kwargs=None, provider_api_keys=None):
    tm = MagicMock()
    tm.task_config = task_config
    tm.turn_based_conversation = False
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.language = None
    tm.language_switcher = None
    tm.kwargs = kwargs or {}
    tm.provider_api_keys = provider_api_keys or {}
    tm.fallback_events = []
    tm.conversation_start_init_ts = 0
    bind(
        tm,
        "_TaskManager__pool_leg_kwargs",
        "_TaskManager__setup_transcriber_with_fallbacks",
        "_TaskManager__build_transcriber_leg",
        "_TaskManager__transcriber_fallback_entries",
        "_TaskManager__add_synthesizer_fallbacks",
        "_TaskManager__build_synthesizer_leg",
        "_TaskManager__synthesizer_fallback_entries",
        "_TaskManager__set_synthesizer_identity",
        "record_fallback",
    )
    return tm


def test_a_single_transcriber_with_fallbacks_runs_as_a_one_language_pool_with_no_language_detector(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tmmod,
        "SUPPORTED_TRANSCRIBER_PROVIDERS",
        {"deepgram": recorder(calls, "deepgram"), "sarvam": recorder(calls, "sarvam")},
    )
    pool = MagicMock()
    monkeypatch.setattr(tmmod, "TranscriberPool", pool)
    tm = setup_tm(
        {
            "tools_config": {
                "input": {"provider": "plivo"},
                "transcriber": {
                    "provider": "deepgram",
                    "model": "nova-3",
                    "language": "hi",
                    "fallbacks": [{"provider": "sarvam", "model": "saaras:v3", "language": None}],
                },
            }
        },
        kwargs={"transcriber_key": "customer-deepgram-key"},
    )

    TaskManager._TaskManager__setup_transcriber.__get__(tm, TaskManager)()

    pool_kwargs = pool.call_args.kwargs
    assert pool_kwargs["active_label"] == "hi"
    assert pool_kwargs.get("lid_provider") is None
    assert [provider for provider, _ in calls] == ["deepgram"]
    assert calls[0][1]["transcriber_key"] == "customer-deepgram-key"

    (entry,) = pool_kwargs["fallbacks"]["hi"]
    assert (entry.provider, entry.model) == ("sarvam", "saaras:v3")
    entry.build()
    provider, built = calls[1]
    assert provider == "sarvam"
    assert built["language"] == "hi"
    assert "transcriber_key" not in built
    assert "fallbacks" not in calls[0][1]


def test_a_single_voice_with_fallbacks_runs_as_a_one_voice_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tmmod,
        "SUPPORTED_SYNTHESIZER_MODELS",
        {"cartesia": recorder(calls, "cartesia"), "sarvam": recorder(calls, "sarvam")},
    )
    pool = MagicMock()
    monkeypatch.setattr(tmmod, "SynthesizerPool", pool)
    sarvam = {
        "provider": "sarvam",
        "provider_config": {"voice": "Anushka", "voice_id": "anushka", "model": "bulbul:v3", "language": "hi"},
    }
    tm = setup_tm(
        {
            "tools_config": {
                "transcriber": {"language": "hi"},
                "synthesizer": {
                    "provider": "cartesia",
                    "provider_config": {"voice": "Riya", "voice_id": "riya-id", "model": "sonic-3", "language": "hi"},
                    "stream": True,
                    "fallbacks": [sarvam],
                },
                "output": {"provider": "plivo"},
                "llm_agent": None,
            }
        },
        kwargs={"synthesizer_key": "customer-cartesia-key"},
    )
    tm.language = "hi"

    TaskManager._TaskManager__setup_synthesizer.__get__(tm, TaskManager)()

    pool_kwargs = pool.call_args.kwargs
    assert list(pool_kwargs["synthesizers"]) == ["hi"]
    assert [provider for provider, _ in calls] == ["cartesia"]
    assert "fallbacks" not in calls[0][1]

    (entry,) = pool_kwargs["fallbacks"]["hi"]
    entry.build()
    provider, built = calls[1]
    assert (provider, built["voice_id"], built["use_mulaw"]) == ("sarvam", "anushka", True)
    assert "synthesizer_key" not in built


def test_a_fallback_takes_its_own_providers_key_when_the_call_has_one():
    tm = setup_tm({}, kwargs={"transcriber_key": "deepgram-key"}, provider_api_keys={"sarvam": "sarvam-key"})
    leg_kwargs = TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, TaskManager)
    assert leg_kwargs(tm.kwargs, "transcriber_key", "sarvam", fallback=True)["transcriber_key"] == "sarvam-key"


async def test_an_llm_switch_is_recorded_and_the_call_follows_the_model_now_serving(monkeypatch):
    tm = setup_tm({})
    tm.llm_agent_config = {"fallbacks": [{"provider": "openai", "model": "gpt-4.1-mini", "reasoning_effort": None}]}
    tm.llm_config = {"provider": "azure", "model": "azure/gpt-4.1-mini", "max_tokens": 200, "temperature": 0.1}
    tm.kwargs = {"llm_key": "azure-key", "llm_fallback_credentials": [{"llm_key": "openai-eu-key"}]}
    tm.on_turn_usage = tm.on_overflow = AsyncMock()
    tm._report_provider_health = AsyncMock()
    tm._cb_tasks = set()
    bind(tm, "_TaskManager__on_llm_fallback", "_TaskManager__spawn_health_report")
    built = []
    monkeypatch.setitem(tmmod.SUPPORTED_LLM_PROVIDERS, "openai", lambda **kwargs: built.append(kwargs) or MagicMock())

    chain = TaskManager._TaskManager__with_llm_fallbacks.__get__(tm, TaskManager)(MagicMock())
    assert isinstance(chain, FallbackLLM)
    (entry,) = chain._pending
    entry.build()
    assert built[0]["llm_key"] == "openai-eu-key"
    assert built[0]["max_tokens"] == 200

    error = type("RateLimitError", (Exception,), {"status_code": 429})("no capacity")
    tm._TaskManager__on_llm_fallback(error, entry, {"sequence_id": 3, "turn_id": 2})
    await asyncio.sleep(0)

    (event,) = tm.fallback_events
    assert event["component"] == "llm"
    assert event["from"] == {"provider": "azure", "model": "azure/gpt-4.1-mini"}
    assert event["to"] == {"provider": "openai", "model": "gpt-4.1-mini"}
    assert (event["reason"], event["sequence_id"]) == ("http_429", 3)
    tm._report_provider_health.assert_awaited_once_with("llm", "azure", "azure/gpt-4.1-mini", False)
    assert (tm.llm_config["provider"], tm.llm_config["model"]) == ("openai", "gpt-4.1-mini")
    assert tm.on_turn_usage is None and tm.on_overflow is None


async def test_an_asr_switch_is_recorded_with_what_the_new_provider_served_and_the_call_follows_it():
    tm = setup_tm({})
    tm.transcriber_provider = "deepgram"
    tm._cb_transcriber_connect_reported = True
    tm._report_provider_health = AsyncMock()
    tm._cb_tasks = set()
    tm._fallback_replacements = []
    tm.llm_latencies = MagicMock(turn_latencies=[])
    bind(tm, "_TaskManager__on_transcriber_fallback", "_TaskManager__spawn_health_report", "fallback_events_with_usage")
    replacement = MagicMock()
    entry = MagicMock(provider="sarvam", model="saaras:v3")

    tm._TaskManager__on_transcriber_fallback("hi", MagicMock(model="nova-3"), replacement, entry, "HTTP 429")
    await asyncio.sleep(0)
    replacement.billed_audio_s.return_value = 41.25

    (event,) = tm.fallback_events_with_usage()
    assert (event["from"], event["to"]) == (
        {"provider": "deepgram", "model": "nova-3"},
        {"provider": "sarvam", "model": "saaras:v3"},
    )
    assert event["served_usage"] == {"audio_s": 41.25}
    assert tm.transcriber_provider == "sarvam"
    assert tm._cb_transcriber_connect_reported is False
    tm._report_provider_health.assert_awaited_once_with("transcriber", "deepgram", "nova-3", False)


def test_each_llm_fallback_is_credited_with_the_turns_it_answered():
    tm = setup_tm({})
    tm._fallback_replacements = []
    tm.llm_latencies = MagicMock(
        turn_latencies=[
            {"sequence_id": seq, "input_tokens": 100 * seq, "output_tokens": seq, "cached_tokens": 0}
            for seq in (1, 2, 3, 4)
        ]
    )
    bind(tm, "fallback_events_with_usage")
    tm.record_fallback("llm", {}, {"model": "second"}, "http_429", "", meta_info={"sequence_id": 2})
    tm.record_fallback("llm", {}, {"model": "third"}, "http_503", "", meta_info={"sequence_id": 4})

    second, third = tm.fallback_events_with_usage()
    assert second["served_usage"] == {"input_tokens": 500, "output_tokens": 5, "cached_tokens": 0, "turns": 2}
    assert third["served_usage"] == {"input_tokens": 400, "output_tokens": 4, "cached_tokens": 0, "turns": 1}


async def test_a_tts_switch_names_both_voices_as_request_logs_do_and_the_call_speaks_as_the_new_one():
    tm = setup_tm({})
    tm.synthesizer_provider = "cartesia"
    tm._report_provider_health = AsyncMock()
    tm._cb_tasks = set()
    tm._fallback_replacements = []
    tm.llm_latencies = MagicMock(turn_latencies=[])
    bind(tm, "_TaskManager__on_synthesizer_fallback", "_TaskManager__spawn_health_report", "fallback_events_with_usage")
    failed, replacement = MagicMock(), MagicMock()
    failed.get_engine.return_value = "sonic-3"
    replacement.get_engine.return_value = "bulbul:v3"
    config = {"provider": "sarvam", "provider_config": {"voice": "Priya", "voice_id": "priya", "model": "bulbul:v3"}}
    entry = MagicMock(provider="sarvam", model="bulbul:v3", config=config)

    tm._TaskManager__on_synthesizer_fallback("hi", failed, replacement, entry, RuntimeError("HTTP 429"))
    await asyncio.sleep(0)
    replacement.get_synthesized_characters.return_value = 310

    (event,) = tm.fallback_events_with_usage()
    assert (event["from"], event["to"]) == (
        {"provider": "cartesia", "model": "sonic-3"},
        {"provider": "sarvam", "model": "bulbul:v3"},
    )
    assert event["served_usage"] == {"characters": 310}
    assert (tm.synthesizer_provider, tm.synthesizer_voice_id) == ("sarvam", "priya")
