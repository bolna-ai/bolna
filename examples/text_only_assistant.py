import asyncio
from bolna.assistant import Assistant
from bolna.models import LlmAgent, SimpleLlmAgent


async def main():
    assistant = Assistant(name="text_only_agent")

    llm_agent = LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config=SimpleLlmAgent(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.2,
        ),
    )

    # No transcriber/synthesizer; enable a text-only pipeline
    assistant.add_task(
        task_type="conversation",
        llm_agent=llm_agent,
        enable_textual_input=True,
    )

    async for chunk in assistant.execute():
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
