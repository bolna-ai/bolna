import asyncio
from bolna.assistant import Assistant
from bolna.models import (
    Transcriber,
    Synthesizer,
    ElevenLabsConfig,
    LlmAgent,
    SimpleLlmAgent,
)


async def main():
    assistant = Assistant(name="demo_agent")

    # Configure audio input (ASR)
    transcriber = Transcriber(provider="deepgram", model="nova-2", stream=True, language="en")

    # Configure LLM
    llm_agent = LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config=SimpleLlmAgent(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.3,
        ),
    )

    # Configure audio output (TTS)
    synthesizer = Synthesizer(
        provider="elevenlabs",
        provider_config=ElevenLabsConfig(
            voice="George", voice_id="JBFqnCBsd6RMkjVDRZzb", model="eleven_turbo_v2_5"
        ),
        stream=True,
        audio_format="wav",
    )

    # Build a single coherent pipeline: transcriber -> llm -> synthesizer
    assistant.add_task(
        task_type="conversation",
        llm_agent=llm_agent,
        transcriber=transcriber,
        synthesizer=synthesizer,
        enable_textual_input=False,
    )

    # Stream results
    async for chunk in assistant.execute():
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
