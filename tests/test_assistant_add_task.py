"""Assistant.add_task must keep the transcriber it is given (README / examples/simple_assistant.py)."""

from bolna.assistant import Assistant
from bolna.models import ElevenLabsConfig, LlmAgent, Synthesizer, Transcriber


def _llm_agent():
    return LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
    )


def _transcriber():
    return Transcriber(provider="deepgram", model="nova-2", stream=True, language="en")


def test_add_task_keeps_supplied_transcriber():
    assistant = Assistant()
    synthesizer = Synthesizer(
        provider="elevenlabs",
        provider_config=ElevenLabsConfig(voice="George", voice_id="JBFqnCBsd6RMkjVDRZzb", model="eleven_turbo_v2_5"),
        stream=True,
    )
    assistant.add_task(
        task_type="conversation", llm_agent=_llm_agent(), transcriber=_transcriber(), synthesizer=synthesizer
    )

    task = assistant.tasks[0]
    assert task["tools_config"]["transcriber"]["provider"] == "deepgram"
    assert task["tools_config"]["transcriber"]["language"] == "en"
    assert task["toolchain"]["pipelines"] == [["transcriber", "llm", "synthesizer"]]


def test_add_task_keeps_transcriber_without_synthesizer():
    assistant = Assistant()
    assistant.add_task(task_type="conversation", llm_agent=_llm_agent(), transcriber=_transcriber())

    task = assistant.tasks[0]
    assert task["tools_config"]["transcriber"]["provider"] == "deepgram"
    assert task["toolchain"]["pipelines"] == [["transcriber", "llm"]]


def test_add_task_without_transcriber_has_no_transcriber_config():
    assistant = Assistant()
    assistant.add_task(task_type="conversation", llm_agent=_llm_agent(), enable_textual_input=True)

    assert assistant.tasks[0]["tools_config"]["transcriber"] is None
