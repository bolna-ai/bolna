"""With lazy language legs, the detector's signal opens the heard language's voice and the decision closes it.

The open happens where the playback gate is armed (end of the caller's turn) and again after the decide drains the
detector buffer (which also covers idle-flush decides); the decision releases its hold as soon as it decides.
Caller-requested agents never open early: they switch only on an explicit ask."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import bolna.agent_manager.task_manager as tmmod
from bolna.agent_manager.task_manager import TaskManager
from bolna.synthesizer.synthesizer_pool import SynthesizerPool
from bolna.transcriber.transcriber_pool import TranscriberPool


def bind(tm, *names):
    for name in names:
        attr = f"_TaskManager__{name}"
        setattr(tm, attr, getattr(TaskManager, attr).__get__(tm, TaskManager))


def make_synth(lazy_legs=True, labels=("hi", "te")):
    synth = MagicMock(spec=SynthesizerPool)
    synth.lazy_legs = lazy_legs
    synth.labels = list(labels)
    return synth


def spawn_tm(segments, lazy_legs=True, explicit_only=False):
    tm = MagicMock()
    tm.language = "hi"
    tm.task_config = {"tools_config": {}}
    tm.lid_playback_gate = None
    tm.language_switcher = MagicMock()
    tm.language_switcher.explicit_only = explicit_only
    pool = MagicMock(spec=TranscriberPool)
    pool.lid_buffer_segments = MagicMock(return_value=segments)
    pool.labels = ["hi", "te"]
    pool.lid_detection_events = []
    synth = make_synth(lazy_legs)
    tm.tools = {"transcriber": pool, "synthesizer": synth}
    tm.handle_language_switch = AsyncMock()
    tm._TaskManager__buffered_language_evidence = TaskManager._TaskManager__buffered_language_evidence
    bind(tm, "detector_language_mismatch", "arm_lid_playback_gate", "open_detected_language", "record_lid_event")
    return tm, synth


async def spawn(tm):
    task = TaskManager._spawn_language_switch_decision.__get__(tm, TaskManager)("mala samajla nahi", {"sequence_id": 7})
    await asyncio.gather(task, return_exceptions=True)
    return task


async def test_turn_end_signal_opens_the_heard_language_voice_held_by_the_decision():
    tm, synth = spawn_tm([{"lang": "te", "audio_s": 2.0}])
    task = await spawn(tm)
    synth.open.assert_called_once_with("te", holder=task, reason="detector_signal")


async def test_caller_requested_agents_never_open_early():
    tm, synth = spawn_tm([{"lang": "te", "audio_s": 2.0}], explicit_only=True)
    await spawn(tm)
    synth.open.assert_not_called()


async def test_no_early_open_without_lazy_legs():
    tm, synth = spawn_tm([{"lang": "te", "audio_s": 2.0}], lazy_legs=False)
    await spawn(tm)
    synth.open.assert_not_called()


async def test_no_early_open_when_the_detector_hears_the_active_language():
    tm, synth = spawn_tm([{"lang": "hi", "audio_s": 2.0}])
    await spawn(tm)
    synth.open.assert_not_called()


async def test_no_early_open_for_an_acknowledgment_length_fragment():
    tm, synth = spawn_tm([{"lang": "te", "audio_s": 0.3}])
    await spawn(tm)
    synth.open.assert_not_called()


def test_open_skips_a_language_without_both_legs():
    tm, synth = spawn_tm([])
    synth.labels = ["hi"]  # no Telugu voice configured
    TaskManager._TaskManager__open_detected_language.__get__(tm, TaskManager)("holder", "te")
    synth.open.assert_not_called()


def decide_tm(monkeypatch, segments, detected):
    monkeypatch.setenv("LANGUAGE_SWITCH_SETTLE_MS", "0")
    tm = MagicMock()
    tm.task_config = {"tools_config": {"llm_agent": {"agent_type": "graph_agent"}}}
    tm.language = "hi"
    tm.conversation_ended = False
    tm.hangup_triggered = False
    tm.function_call_in_flight = False
    tm.multilingual_prompts = {}
    tm.lid_playback_gate = None
    tm._should_ignore_transcriber_input = MagicMock(return_value=False)
    pool = MagicMock(spec=TranscriberPool)
    pool.labels = ["hi", "mr"]
    pool.lid_detection_events = []
    pool.lid_buffer_max_segment_seconds.return_value = max((s["audio_s"] for s in segments), default=0.0)
    pool.lid_buffer_language_confidence.return_value = 0.9
    pool.lid_buffer_segments.return_value = segments
    pool.take_lid_transcript.return_value = ("kahi tari", detected)
    synth = make_synth(labels=("hi", "mr"))
    tm.tools = {"transcriber": pool, "synthesizer": synth, "input": MagicMock()}
    tm.language_switcher = MagicMock()
    tm.language_switcher.explicit_only = False
    tm.language_switcher.decide = AsyncMock(return_value={"target_language": "hi", "reasoning": "stay"})
    tm._inflight_response_activity = MagicMock(return_value={"audio_playing": False})
    bind(tm, "switch_settle_ms", "switch_decide_timeout_s", "record_lid_event", "open_detected_language")
    return tm, synth


async def test_decide_opens_the_drained_language_voice(monkeypatch):
    # The idle-flush path never passes the turn-end check, so this is its only early open.
    tm, synth = decide_tm(monkeypatch, [{"lang": "mr", "prob": 0.9, "audio_s": 2.0}], "mr")
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    task = asyncio.create_task(run("", None, "hi"))
    await task
    synth.open.assert_called_once_with("mr", holder=task, reason="detector_signal")


async def test_decide_does_not_open_for_a_short_fragment(monkeypatch):
    tm, synth = decide_tm(monkeypatch, [{"lang": "mr", "prob": 0.9, "audio_s": 0.3}], "mr")
    run = TaskManager._TaskManager__run_language_switch.__get__(tm, TaskManager)
    await run("", None, "hi")
    synth.open.assert_not_called()


def handle_tm(run_side_effect=None):
    tm = MagicMock()
    tm.language_switcher.explicit_only = False
    tm.language_switch_lock = asyncio.Lock()
    tm._spec_followup_task = None
    tm._TaskManager__run_language_switch = AsyncMock(return_value=None, side_effect=run_side_effect)
    synth = make_synth()
    tm.tools = {"synthesizer": synth}
    bind(tm, "release_detected_languages")
    return tm, synth


async def test_the_decision_releases_its_hold_once_it_decides():
    tm, synth = handle_tm()
    handle = TaskManager.handle_language_switch.__get__(tm, TaskManager)
    task = asyncio.create_task(handle("garbled", {"sequence_id": 3}, spawn_language="hi"))
    await task
    synth.release.assert_called_once_with(task)


async def test_a_failing_decision_still_releases_its_hold():
    tm, synth = handle_tm(run_side_effect=RuntimeError("judge exploded"))
    handle = TaskManager.handle_language_switch.__get__(tm, TaskManager)
    task = asyncio.create_task(handle("garbled", {"sequence_id": 3}, spawn_language="hi"))
    await task
    synth.release.assert_called_once_with(task)


async def test_release_is_a_noop_without_lazy_legs():
    tm, synth = handle_tm()
    synth.lazy_legs = False
    await TaskManager.handle_language_switch.__get__(tm, TaskManager)("garbled", {}, spawn_language="hi")
    synth.release.assert_not_called()


def test_every_multilingual_call_builds_lazy_pools(monkeypatch):
    # No flag: once deployed, every multilingual call connects only the languages it uses.
    transcriber_pool, synthesizer_pool = MagicMock(), MagicMock()
    monkeypatch.setattr(tmmod, "TranscriberPool", transcriber_pool)
    monkeypatch.setattr(tmmod, "SynthesizerPool", synthesizer_pool)
    monkeypatch.setattr(tmmod, "SUPPORTED_TRANSCRIBER_PROVIDERS", {"deepgram": MagicMock()})
    monkeypatch.setattr(tmmod, "SUPPORTED_SYNTHESIZER_MODELS", {"elevenlabs": MagicMock()})
    tm = MagicMock()
    tm.turn_based_conversation = True
    tm.is_web_based_call = False
    tm.enforce_streaming = True
    tm.kwargs = {}
    tm.provider_api_keys = {}
    tm._TaskManager__pool_leg_kwargs = TaskManager._TaskManager__pool_leg_kwargs.__get__(tm, TaskManager)
    tm._TaskManager__language_switch_enabled = MagicMock(return_value=False)
    voice = {"provider": "elevenlabs", "provider_config": {"voice": "George"}}
    tm.task_config = {
        "tools_config": {
            "transcriber": {
                "provider": "deepgram",
                "language": "en",
                "multilingual": {"en": {"provider": "deepgram"}, "hi": {"provider": "deepgram"}},
            },
            "synthesizer": {"provider": "elevenlabs", "multilingual": {"en": dict(voice), "hi": dict(voice)}},
            "output": {"provider": "plivo"},
            "llm_agent": None,
        }
    }

    TaskManager._TaskManager__setup_transcriber.__get__(tm, TaskManager)()
    TaskManager._TaskManager__setup_synthesizer.__get__(tm, TaskManager)()

    assert transcriber_pool.call_args.kwargs["lazy_legs"] is True
    assert synthesizer_pool.call_args.kwargs["lazy_legs"] is True
